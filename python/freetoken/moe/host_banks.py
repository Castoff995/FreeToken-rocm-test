"""Reusable pinned host-bank primitives shared by the fast expert-load paths.

Two ideas the parallel read of the original checkpoint and FTW (read a repacked
contiguous cache) paths both rely on:

* **pin-after-fill** -- allocate the bank as a *lazy* anonymous ``mmap`` (no pages
  resident, instant), fill it with real data, and only THEN ``cudaHostRegister`` it.
  Registering already-resident pages just page-locks them; registering a lazy mmap first
  faults+zero-fills every page (~137 GiB -> ~47 s for DSV4) and that zero-fill is then
  immediately overwritten by the read. So pin-after-fill removes a whole redundant pass.
* **chunked multi-threaded O_DIRECT** -- DMA straight from disk into the (page-aligned)
  bank, bypassing the page cache, with many concurrent ``preadv`` on one fd (scales to the
  device's queue-depth ceiling even for a single file).

The mmaps are held for the process lifetime (the banks live as long as the offload cache).
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import torch

from freetoken.utils import init_logger


logger = init_logger(__name__)

_BLK = 4096  # O_DIRECT alignment (page size)


class HostResidency(str, Enum):
    """Residency class of a host bank layer.

    ``PINNED`` (cudaHostRegister'd) is required for anything the GPU dereferences:
    the decode gather kernels and pointer-based prefill copies. ``PAGEABLE`` is served
    through ordinary PyTorch H2D copies on Windows ROCm when registration fails.
    ``LOCKED`` (VirtualLock/mlock: resident for the CPU executor, but with no device
    address) remains reserved for CPU-executor policies that are not implemented here.
    """

    PINNED = "pinned"
    LOCKED = "locked"
    PAGEABLE = "pageable"


_DEFAULT_CHUNK = 8 << 20
_HOST_RESIDENCY_ATTR = "_freetoken_host_residency"
_HOST_PIN_ERROR_ATTR = "_freetoken_host_pin_error"

# Hold the mmaps for the process lifetime; the offload cache reads from these banks forever.
_LIVE_BUFFERS: list[mmap.mmap] = []


class HostBank:
    """A lazily-allocated, page-aligned host buffer + its torch view, registered on demand.

    Allocate -> fill (write real data, or O_DIRECT read into it) -> ``pin()``. The mmap is
    rounded up to the O_DIRECT block so chunked reads are always aligned; ``tensor`` views
    exactly ``nbytes``."""

    __slots__ = ("tensor", "addr", "nbytes", "_buf", "_pinned", "_pin_error")

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        elsize = torch.empty((), dtype=dtype).element_size()
        self.nbytes = math.prod(shape) * elsize
        asize = ((self.nbytes + _BLK - 1) // _BLK) * _BLK
        self._buf = mmap.mmap(-1, asize)  # lazy: address space only, no resident pages yet
        _LIVE_BUFFERS.append(self._buf)
        self.addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buf))
        self.tensor = torch.frombuffer(self._buf, dtype=dtype, count=self.nbytes // elsize).view(*shape)
        self._pinned = False
        self._pin_error: str | None = None
        setattr(self.tensor, _HOST_RESIDENCY_ATTR, HostResidency.PAGEABLE.value)
        setattr(self.tensor, _HOST_PIN_ERROR_ATTR, None)

    @property
    def residency(self) -> HostResidency:
        return HostResidency.PINNED if self._pinned else HostResidency.PAGEABLE

    def memoryview(self) -> memoryview:
        return memoryview(self._buf)

    def pin(self) -> None:
        """Register and map the now-filled resident buffer -- pin-after-fill."""
        if self._pinned:
            return
        from freetoken.kernel.pinned import host_register

        try:
            host_register(self.addr, len(self._buf))
        except RuntimeError as exc:
            self._pin_error = str(exc)
            setattr(self.tensor, _HOST_RESIDENCY_ATTR, HostResidency.PAGEABLE.value)
            setattr(self.tensor, _HOST_PIN_ERROR_ATTR, self._pin_error)
            raise RuntimeError(
                f"host registration failed for {len(self._buf) / 2**30:.1f} GiB"
            ) from exc
        self._pinned = True
        self._pin_error = None
        setattr(self.tensor, _HOST_RESIDENCY_ATTR, HostResidency.PINNED.value)
        setattr(self.tensor, _HOST_PIN_ERROR_ATTR, None)

    def release(self) -> None:
        """Drop the buffer's resident pages (address space stays valid; contents
        become undefined). Only for buffers that are done being read -- the
        converter releases each layer after writing it out."""
        assert not self._pinned, "cannot release a pinned bank"
        self._buf.madvise(mmap.MADV_DONTNEED)

    def lock(self) -> None:
        """Make the buffer resident without a device mapping (VirtualLock/mlock).

        Platform-specific and not implemented here. A locked bank layer is
        CPU-executor-readable but must never be handed to the GPU movement
        paths.
        """
        raise NotImplementedError("HostBank.lock() is platform-specific and not implemented")


def alloc_banks(specs: dict[str, tuple[tuple[int, ...], torch.dtype]]) -> dict[str, HostBank]:
    """Allocate (lazy, unpinned) host banks from ``{name: (shape, dtype)}``."""
    return {name: HostBank(shape, dtype) for name, (shape, dtype) in specs.items()}


def alloc_layer_banks(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], num_layers: int
) -> dict[str, list[HostBank]]:
    """Allocate per-layer host banks: ``{name: ([num_experts, ...] row shape, dtype)}``
    -> one independently allocated (page-aligned, independently pin/lock-able)
    ``HostBank`` per layer per name."""
    return {
        name: [HostBank(shape, dtype) for _ in range(num_layers)]
        for name, (shape, dtype) in specs.items()
    }


def pin_banks(banks: dict[str, HostBank | list[HostBank]]) -> None:
    """Pin (register) every bank after it has been filled -- pin-after-fill."""
    for bank in banks.values():
        if isinstance(bank, list):
            for layer_bank in bank:
                layer_bank.pin()
        else:
            bank.pin()


def _windows_rocm_pageable_fallback() -> bool:
    return sys.platform == "win32" and bool(getattr(torch.version, "hip", None))


def infer_host_layer_residency(
    sources: dict[str, list[torch.Tensor]],
) -> list[str] | None:
    """Infer per-layer residency from tensors backed by :class:`HostBank`.

    ``None`` preserves the historical default for providers that do not expose
    HostBank metadata. Once any source exposes metadata, a layer is considered pinned
    only when every required bank explicitly reports successful registration.
    """

    if not sources:
        return None
    num_layers = len(next(iter(sources.values())))
    values = [
        getattr(source, _HOST_RESIDENCY_ATTR, None)
        for per_layer in sources.values()
        for source in per_layer
    ]
    if not any(value is not None for value in values):
        return None

    residency: list[str] = []
    for layer_id in range(num_layers):
        states = [
            getattr(per_layer[layer_id], _HOST_RESIDENCY_ATTR, None)
            for per_layer in sources.values()
        ]
        if all(state == HostResidency.PINNED.value for state in states):
            residency.append(HostResidency.PINNED.value)
        else:
            residency.append(HostResidency.PAGEABLE.value)
    return residency


class PinPipeline:
    """Pin filled banks while other banks are still being read.

    cudaHostRegister is driver-serialized, so one background thread drains a
    queue of completed banks; submitters never block. Total load time becomes
    ~max(read, pin) instead of their sum. Use as a context manager: a clean exit
    drains the queue and re-raises the first pin failure, except that Windows ROCm
    registration failures retain that bank as pageable and continue. An exceptional
    exit still joins the thread but lets the original exception propagate.
    """

    def __init__(self) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._exc: BaseException | None = None
        self._mapped_count = 0
        self._pageable_count = 0
        self._first_fallback: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            bank = self._q.get()
            if bank is None:
                return
            if self._exc is not None:
                continue  # drain without pinning after a failure
            try:
                bank.pin()
                self._mapped_count += 1
            except BaseException as exc:  # surfaced by wait()/__exit__
                if _windows_rocm_pageable_fallback() and isinstance(exc, RuntimeError):
                    # Windows/WDDM may reject an individual hipHostRegister even after
                    # earlier banks succeeded. Keep this bank as ordinary host memory;
                    # residency-aware cache dispatch will stage its whole layer through
                    # PyTorch H2D and must never expose its raw CPU VA to a GPU kernel.
                    self._pageable_count += 1
                    if self._first_fallback is None:
                        self._first_fallback = exc
                    continue
                self._exc = exc

    def submit(self, bank: HostBank) -> None:
        self._q.put(bank)

    def __call__(self, layer_id: int, banks: dict[str, HostBank]) -> None:
        """Layer-completion sink: queue every bank of the completed layer."""
        for bank in banks.values():
            self.submit(bank)

    def _join(self) -> None:
        self._q.put(None)
        self._thread.join()

    def wait(self) -> None:
        self._join()
        if self._exc is not None:
            raise self._exc
        if self._pageable_count:
            first = self._first_fallback
            reason = first.__cause__ if first is not None and first.__cause__ else first
            logger.warning(
                "Windows ROCm host mapping fallback: "
                f"mapped_banks={self._mapped_count} "
                f"pageable_banks={self._pageable_count} "
                f"first_failure={reason}"
            )

    def __enter__(self) -> "PinPipeline":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._join()  # no thread leak; the in-flight exception wins
            return
        self.wait()


class LayerCompletionTracker:
    """Fire a sink once per layer, when all of that layer's writes have landed.

    ``note(layer_id)`` is called after each write; at ``expected_per_layer``
    notes the layer's banks are handed to ``on_layer(layer_id, {name: bank})``
    exactly once. Thread-safe (shard-driven loaders write layers from many
    threads in arbitrary order).
    """

    def __init__(
        self,
        expected_per_layer: int,
        banks: dict[str, list],
        on_layer,
    ) -> None:
        assert expected_per_layer > 0
        self._expected = expected_per_layer
        self._banks = banks
        self._on_layer = on_layer
        self._counts: dict[int, int] = {}
        self._lock = threading.Lock()

    def note(self, layer_id: int) -> None:
        with self._lock:
            n = self._counts.get(layer_id, 0) + 1
            self._counts[layer_id] = n
            fire = n == self._expected
        if fire:
            self._on_layer(layer_id, {name: per[layer_id] for name, per in self._banks.items()})


def read_file_into(buf: memoryview | mmap.mmap, path: str, *, workers: int = 8,
                   chunk: int = _DEFAULT_CHUNK, drop_cache: bool = True) -> int:
    """Chunked multi-threaded O_DIRECT read of the whole file ``path`` into ``buf``
    (page-aligned). Returns the file size. The buffer must be >= the rounded-up file size."""
    size = os.path.getsize(path)
    if drop_cache:
        try:
            fd0 = os.open(path, os.O_RDONLY)
            os.posix_fadvise(fd0, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd0)
        except OSError:
            pass
    mv = buf if isinstance(buf, memoryview) else memoryview(buf)
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    offs = list(range(0, size, chunk))

    def rd(o):
        want = min(chunk, len(mv) - o)
        want = min(want, ((size - o + _BLK - 1) // _BLK) * _BLK)
        os.preadv(fd, [mv[o:o + want]], o)

    try:
        if len(offs) <= 1:
            for o in offs:
                rd(o)
        else:
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(rd, offs))
    finally:
        os.close(fd)
    return size


__all__ = [
    "HostBank",
    "HostResidency",
    "LayerCompletionTracker",
    "PinPipeline",
    "alloc_banks",
    "alloc_layer_banks",
    "infer_host_layer_residency",
    "pin_banks",
    "read_file_into",
]
