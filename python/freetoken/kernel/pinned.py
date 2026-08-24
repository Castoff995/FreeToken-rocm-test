"""Exact-size pinned host tensors (e.g. offload expert banks).

The offload gather kernel (``fast_index_copy``) reads host memory zero-copy from the
GPU, so allocations must be pinned + device-mapped. We avoid
``torch.empty(pin_memory=True)`` because its caching allocator rounds sizes up to the
next power of two (a 70GB bank would reserve 128GB)."""

from __future__ import annotations

import ctypes
import importlib
import mmap
import os
from dataclasses import dataclass
from functools import lru_cache

import torch


_HIP_HOST_REGISTER_PORTABLE = 1
_HIP_HOST_REGISTER_MAPPED = 2
_HIP_HOST_REGISTER_FLAGS = _HIP_HOST_REGISTER_PORTABLE | _HIP_HOST_REGISTER_MAPPED
_HIP_PROBE_BYTES = 4096
_HIP_LIBRARY_NAMES = ("amdhip64_7.dll", "amdhip64.dll", "libamdhip64.so")


@dataclass(frozen=True)
class HostMappingResult:
    device_ptr: int | None
    mapping_backend: str
    extension_loaded: bool
    reason: str

    @property
    def available(self) -> bool:
        return self.device_ptr is not None and self.device_ptr > 0


@lru_cache(maxsize=1)
def _load_pinned_extension():
    try:
        return importlib.import_module("freetoken.kernel._pinned_tensor")
    except ImportError:
        # patched: ROCm/Windows fallback -- no exact-size pinned extension;
        # callers fall back to torch's own pinned allocator (rounds sizes up).
        return None


def _configure_hip_runtime(runtime):
    """Set exact ctypes signatures for the HIP host-mapping APIs we call."""

    runtime.hipHostRegister.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    runtime.hipHostRegister.restype = ctypes.c_int
    runtime.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    runtime.hipHostGetDevicePointer.restype = ctypes.c_int
    runtime.hipHostUnregister.argtypes = [ctypes.c_void_p]
    runtime.hipHostUnregister.restype = ctypes.c_int
    return runtime


def _hip_runtime_candidates():
    seen: set[str] = set()
    for name in _HIP_LIBRARY_NAMES:
        seen.add(name)
        yield name

    for env_name in ("ROCM_HOME", "HIP_PATH"):
        root = os.getenv(env_name)
        if not root:
            continue
        for subdir in ("bin", "lib", "lib64", ""):
            for name in _HIP_LIBRARY_NAMES:
                candidate = (
                    os.path.join(root, subdir, name)
                    if subdir
                    else os.path.join(root, name)
                )
                if candidate not in seen:
                    seen.add(candidate)
                    yield candidate


@lru_cache(maxsize=1)
def _load_hip_runtime():
    """Load the HIP runtime only in a ROCm PyTorch process."""

    if not getattr(torch.version, "hip", None):
        return None
    for candidate in _hip_runtime_candidates():
        try:
            return _configure_hip_runtime(ctypes.CDLL(candidate))
        except (AttributeError, OSError):
            continue
    return None


def _hip_host_register(runtime, addr: int, nbytes: int) -> int:
    return int(
        runtime.hipHostRegister(
            ctypes.c_void_p(addr),
            ctypes.c_size_t(nbytes),
            ctypes.c_uint(_HIP_HOST_REGISTER_FLAGS),
        )
    )


def _hip_host_device_ptr(runtime, addr: int) -> int:
    mapped = ctypes.c_void_p()
    status = int(
        runtime.hipHostGetDevicePointer(
            ctypes.byref(mapped),
            ctypes.c_void_p(addr),
            ctypes.c_uint(0),
        )
    )
    if status != 0:
        raise RuntimeError(f"hipHostGetDevicePointer failed with hipError {status}")
    if not mapped.value:
        raise RuntimeError("hipHostGetDevicePointer returned a null device pointer")
    return int(mapped.value)


def resolve_host_mapping(addr: int) -> HostMappingResult:
    """Resolve a host address without ever treating the raw CPU VA as proof.

    The packaged extension is preferred. ROCm processes fall back to the HIP runtime;
    any missing API, exception, failed status, or null pointer reports an unavailable
    mapping so callers can select staged H2D copies.
    """

    if addr <= 0:
        return HostMappingResult(None, "unavailable", False, "host_pointer_invalid")

    try:
        extension = _load_pinned_extension()
    except Exception:
        extension = None
        extension_load_failed = True
    else:
        extension_load_failed = False

    if extension is not None:
        resolver = getattr(extension, "host_device_ptr", None)
        if not callable(resolver):
            return HostMappingResult(
                None,
                "unavailable",
                True,
                "host_device_ptr_unavailable",
            )
        try:
            mapped = int(resolver(addr))
        except Exception:
            return HostMappingResult(
                None,
                "unavailable",
                True,
                "host_device_mapping_failed",
            )
        if mapped <= 0:
            return HostMappingResult(
                None,
                "unavailable",
                True,
                "host_device_pointer_zero",
            )
        return HostMappingResult(mapped, "extension", True, "ok")

    try:
        runtime = _load_hip_runtime()
    except Exception:
        runtime = None
    if runtime is None:
        reason = (
            "pinned_extension_load_failed"
            if extension_load_failed
            else "mapping_backend_unavailable"
        )
        return HostMappingResult(None, "unavailable", False, reason)
    try:
        mapped = _hip_host_device_ptr(runtime, addr)
    except Exception:
        return HostMappingResult(
            None,
            "unavailable",
            False,
            "host_device_mapping_failed",
        )
    return HostMappingResult(mapped, "hip_runtime", False, "ok")


def create_pinned_tensor_like(input: torch.Tensor) -> torch.Tensor:
    """Create a CPU pinned tensor with the same size, stride, and dtype as input."""
    ext = _load_pinned_extension()
    if ext is None:
        out = torch.empty_like(input, pin_memory=True)
        return out
    return ext.create_pinned_tensor_like(input)


def copy_to_pinned_tensor(input: torch.Tensor) -> torch.Tensor:
    """Copy a CPU tensor into exact-size cudaMallocHost pinned storage."""

    output = create_pinned_tensor_like(input)
    with torch.no_grad():
        output.copy_(input)
    return output


def alloc_pinned_tensor(*shape: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate an exact-size, uninitialized pinned host tensor via cudaHostAlloc."""
    ext = _load_pinned_extension()
    if ext is None:
        return torch.empty(*shape, dtype=dtype, pin_memory=True)
    return ext.alloc_pinned_tensor(list(shape), dtype)


def host_register(addr: int, nbytes: int) -> None:
    """Register an existing nonempty host range as portable and device-mapped.

    Zero-length ranges are rejected rather than passed to CUDA/HIP with ambiguous
    semantics. The packaged extension remains first choice; ROCm without it calls the
    HIP runtime directly. Expert banks intentionally remain registered for their process
    lifetime.
    """

    if addr <= 0:
        raise ValueError(f"host registration address must be nonzero, got {addr}")
    if nbytes <= 0:
        raise ValueError(f"host registration size must be positive, got {nbytes}")
    ext = _load_pinned_extension()
    if ext is not None:
        ext.host_register(addr, nbytes)
        return
    if getattr(torch.version, "hip", None):
        runtime = _load_hip_runtime()
        if runtime is None:
            raise RuntimeError("HIP runtime host-mapping APIs are unavailable")
        status = _hip_host_register(runtime, addr, nbytes)
        if status != 0:
            raise RuntimeError(
                f"hipHostRegister({nbytes} bytes) failed with hipError {status}"
            )


@lru_cache(maxsize=1)
def _host_ptr_identity() -> bool:
    # cached per process: FreeToken pins one CUDA device per process (set at engine launch)
    ext = _load_pinned_extension()
    if ext is not None:
        return bool(ext.host_ptr_identity())
    runtime = _load_hip_runtime()
    if runtime is None:
        return False

    # Probe the same registration mode expert-bank mmaps use. On Windows/WDDM,
    # torch pin_memory allocations may have identity mappings while hipHostRegister'd
    # mmap memory has a distinct device alias.
    buffer = mmap.mmap(-1, _HIP_PROBE_BYTES)
    exported = None
    registered = False
    try:
        exported = ctypes.c_char.from_buffer(buffer)
        addr = ctypes.addressof(exported)
        if _hip_host_register(runtime, addr, _HIP_PROBE_BYTES) != 0:
            return False
        registered = True
        try:
            mapped = _hip_host_device_ptr(runtime, addr)
        except RuntimeError:
            return False
        return mapped == addr
    finally:
        cleanup_error: BaseException | None = None
        if registered:
            try:
                status = int(runtime.hipHostUnregister(ctypes.c_void_p(addr)))
                if status != 0:
                    cleanup_error = RuntimeError(
                        f"hipHostUnregister probe cleanup failed with hipError {status}"
                    )
            except BaseException as exc:
                cleanup_error = exc
        if exported is not None:
            del exported
        try:
            buffer.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if cleanup_error is not None:
            raise cleanup_error


def device_ptr(t: torch.Tensor) -> int:
    """Base address of ``t`` as the GPU must dereference it.

    Equals ``data_ptr()`` on CUDA tensors and wherever pinned host memory is
    device-visible at its host VA (Linux/UVA). On Windows/WDDM registered memory maps
    to a different device address, so zero-copy consumers must use this, not
    ``data_ptr()``. Host tensors must be pinned+mapped."""
    if t.is_cuda:
        return t.data_ptr()
    mapping = resolve_host_mapping(int(t.data_ptr()))
    if mapping.available:
        return int(mapping.device_ptr)
    if (
        mapping.extension_loaded
        or mapping.reason == "pinned_extension_load_failed"
        or getattr(torch.version, "hip", None)
    ):
        raise RuntimeError(
            "host tensor has no demonstrable GPU-visible mapping "
            f"(reason={mapping.reason})"
        )
    # Preserve the historical non-ROCm/no-extension behavior. Offload GPU-deref
    # consumers only use this path where their platform capability permits it.
    return t.data_ptr()


__all__ = [
    "HostMappingResult",
    "alloc_pinned_tensor",
    "copy_to_pinned_tensor",
    "create_pinned_tensor_like",
    "device_ptr",
    "host_register",
    "resolve_host_mapping",
]
