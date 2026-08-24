from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from freetoken.kernel import pinned


@pytest.fixture(autouse=True)
def _clear_pinned_caches():
    pinned._load_pinned_extension.cache_clear()
    pinned._load_hip_runtime.cache_clear()
    pinned._host_ptr_identity.cache_clear()
    yield
    pinned._load_pinned_extension.cache_clear()
    pinned._load_hip_runtime.cache_clear()
    pinned._host_ptr_identity.cache_clear()


def _runtime(*, register_status: int = 0, mapped_ptr: int = 0xABCDEF00):
    runtime = SimpleNamespace()
    runtime.hipGetLastError = mock.Mock(return_value=register_status)
    runtime.hipHostRegister = mock.Mock(return_value=register_status)

    def get_device_pointer(output, _host, _flags):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = mapped_ptr
        return 0

    runtime.hipHostGetDevicePointer = mock.Mock(side_effect=get_device_pointer)
    runtime.hipHostUnregister = mock.Mock(return_value=0)
    return runtime


def test_extension_takes_precedence_over_hip_runtime():
    extension = mock.Mock()
    extension.host_device_ptr.return_value = 0x2468
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=extension),
        mock.patch.object(
            pinned,
            "_load_hip_runtime",
            side_effect=AssertionError("HIP runtime must not be queried"),
        ) as runtime_loader,
    ):
        pinned.host_register(0x1234, 4096)
        result = pinned.resolve_host_mapping(0x1234)

    extension.host_register.assert_called_once_with(0x1234, 4096)
    extension.host_device_ptr.assert_called_once_with(0x1234)
    runtime_loader.assert_not_called()
    assert result == pinned.HostMappingResult(0x2468, "extension", True, "ok")


def test_hip_runtime_is_not_loaded_outside_rocm():
    with (
        mock.patch.object(torch.version, "hip", None),
        mock.patch.object(
            pinned.ctypes,
            "CDLL",
            side_effect=AssertionError("no runtime load expected"),
        ) as loader,
    ):
        assert pinned._load_hip_runtime() is None
    loader.assert_not_called()


def test_hip_function_signatures_are_configured_explicitly():
    runtime = _runtime()

    assert pinned._configure_hip_runtime(runtime) is runtime
    assert runtime.hipGetLastError.argtypes == []
    assert runtime.hipGetLastError.restype is ctypes.c_int
    assert runtime.hipHostRegister.argtypes == [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    assert runtime.hipHostRegister.restype is ctypes.c_int
    assert runtime.hipHostGetDevicePointer.argtypes == [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    assert runtime.hipHostGetDevicePointer.restype is ctypes.c_int
    assert runtime.hipHostUnregister.argtypes == [ctypes.c_void_p]
    assert runtime.hipHostUnregister.restype is ctypes.c_int


def test_hip_registration_uses_portable_and_mapped_flags():
    runtime = _runtime()
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
        mock.patch.object(torch.version, "hip", "test-rocm"),
    ):
        pinned.host_register(0x12340000, 8192)

    address, nbytes, flags = runtime.hipHostRegister.call_args.args
    assert address.value == 0x12340000
    assert nbytes.value == 8192
    assert flags.value == (
        pinned._HIP_HOST_REGISTER_PORTABLE | pinned._HIP_HOST_REGISTER_MAPPED
    )
    runtime.hipGetLastError.assert_not_called()


def test_zero_byte_registration_is_rejected_without_calling_hip():
    runtime = _runtime()
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        pytest.raises(ValueError, match="size must be positive"),
    ):
        pinned.host_register(0x12340000, 0)
    runtime.hipHostRegister.assert_not_called()


def test_hip_registration_failure_is_surfaced():
    runtime = _runtime(register_status=17)
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        pytest.raises(RuntimeError, match="hipError 17"),
    ):
        pinned.host_register(0x12340000, 4096)
    runtime.hipGetLastError.assert_not_called()


def test_direct_registration_failure_can_be_cleared_without_losing_status():
    runtime = _runtime(register_status=17)
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        pytest.raises(RuntimeError, match="hipError 17") as failure,
    ):
        pinned.host_register(0x12340000, 4096)

    assert pinned._clear_recoverable_hip_host_register_error(failure.value) == 17
    runtime.hipGetLastError.assert_called_once_with()
    assert "hipError 17" in str(failure.value)


def test_hip_mapping_returns_translated_nonidentity_pointer():
    runtime = _runtime(mapped_ptr=0xFEDCBA00)
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
    ):
        result = pinned.resolve_host_mapping(0x12340000)

    assert result.available
    assert result.device_ptr == 0xFEDCBA00
    assert result.device_ptr != 0x12340000
    assert result.mapping_backend == "hip_runtime"
    assert not result.extension_loaded


def test_mapping_failure_never_proves_raw_host_pointer():
    runtime = _runtime()
    runtime.hipHostGetDevicePointer.side_effect = lambda *_args: 11
    host_ptr = 0x12340000
    with (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
    ):
        result = pinned.resolve_host_mapping(host_ptr)

    assert not result.available
    assert result.device_ptr is None
    assert result.device_ptr != host_ptr
    assert result.mapping_backend == "unavailable"
    assert result.reason == "host_device_mapping_failed"


def test_device_ptr_raises_instead_of_returning_raw_pointer_on_rocm_failure():
    tensor = torch.empty(4, dtype=torch.uint8)
    unavailable = pinned.HostMappingResult(
        None, "unavailable", False, "host_device_mapping_failed"
    )
    with (
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch.object(pinned, "resolve_host_mapping", return_value=unavailable),
        pytest.raises(RuntimeError, match="no demonstrable GPU-visible mapping"),
    ):
        pinned.device_ptr(tensor)


class _ProbeBuffer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _probe_patches(runtime, buffer, *, mapped_side_effect):
    fake_char = SimpleNamespace(from_buffer=mock.Mock(return_value=object()))
    return (
        mock.patch.object(pinned, "_load_pinned_extension", return_value=None),
        mock.patch.object(pinned, "_load_hip_runtime", return_value=runtime),
        mock.patch.object(pinned.mmap, "mmap", return_value=buffer),
        mock.patch.object(pinned.ctypes, "c_char", fake_char),
        mock.patch.object(pinned.ctypes, "addressof", return_value=0x45670000),
        mock.patch.object(pinned, "_hip_host_register", return_value=0),
        mock.patch.object(pinned, "_hip_host_device_ptr", side_effect=mapped_side_effect),
    )


def test_identity_probe_unregisters_and_closes_on_success():
    runtime = _runtime()
    buffer = _ProbeBuffer()
    patches = _probe_patches(runtime, buffer, mapped_side_effect=lambda *_: 0x45670000)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert pinned._host_ptr_identity()

    unregister_address = runtime.hipHostUnregister.call_args.args[0]
    assert unregister_address.value == 0x45670000
    assert buffer.closed


def test_identity_probe_unregisters_and_closes_after_mapping_failure():
    runtime = _runtime()
    buffer = _ProbeBuffer()
    patches = _probe_patches(
        runtime,
        buffer,
        mapped_side_effect=RuntimeError("mapping unavailable"),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
        assert not pinned._host_ptr_identity()

    runtime.hipHostUnregister.assert_called_once()
    assert buffer.closed
