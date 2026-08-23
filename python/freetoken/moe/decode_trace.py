"""Opt-in diagnostics for the first real autoregressive decode forward.

The trace is deliberately process-local and one-shot.  It is claimed by
``Engine.forward_batch`` only for a real ``Batch(phase="decode")``; startup prefill
warmup therefore cannot consume it.  Normal execution pays only an environment
check and an inactive-state branch.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Any

import torch


TRACE_ENV = "FREETOKEN_ROCM_DECODE_TRACE"
TRACE_PREFIX = "FT_ROCM_DECODE_TRACE"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass
class _TraceState:
    claimed: bool = False
    active: bool = False
    sequence: int = 0


_state = _TraceState()


@dataclass(frozen=True)
class _HostMappingStatus:
    safe_for_gpu_deref: bool
    pinned_extension_loaded: bool
    host_bank_residency: str
    bank_count: int
    mapped_bank_count: int
    host_ptr: str
    device_ptr: str
    failed_bank: str
    reason: str


def trace_enabled() -> bool:
    return os.getenv(TRACE_ENV, "").strip().lower() in _TRUE_VALUES


def trace_active() -> bool:
    return _state.active


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_render(item) for item in value) + "]"
    text = str(value)
    return repr(text) if any(ch.isspace() for ch in text) else text


def trace_stage(stage: str, /, **fields: Any) -> None:
    """Emit one stable, flush-on-write marker while the first-decode trace is active."""

    if not _state.active:
        return
    _state.sequence += 1
    parts = [TRACE_PREFIX, f"seq={_state.sequence}", f"stage={stage}"]
    parts.extend(f"{key}={_render(fields[key])}" for key in sorted(fields))
    print(" ".join(parts), file=sys.stderr, flush=True)


def _stream_fields(device: torch.device | str | None = None) -> dict[str, Any]:
    try:
        if not torch.cuda.is_available():
            return {"stream": "unavailable"}
        stream = torch.cuda.current_stream(device)
        return {
            "stream": int(stream.cuda_stream),
            "stream_device": str(stream.device),
        }
    except Exception as exc:  # diagnostics must not mask the stage being diagnosed
        return {"stream": f"unavailable:{type(exc).__name__}"}


def begin_first_decode(batch: Any) -> bool:
    """Claim and begin the first real decode batch in this process, if requested."""

    if _state.claimed or not trace_enabled() or not bool(getattr(batch, "is_decode", False)):
        return False
    _state.claimed = True
    _state.active = True
    batch_size = getattr(batch, "size", None)
    fields: dict[str, Any] = {
        "batch_size": batch_size if batch_size is not None else "unknown",
        "phase": getattr(batch, "phase", "unknown"),
    }
    fields.update(_stream_fields())
    trace_stage("decode_entry", **fields)
    input_ids = getattr(batch, "input_ids", None)
    if isinstance(input_ids, torch.Tensor):
        trace_tensor("decode_input_ids", input_ids, include_minmax=True)
    return True


def finish_trace(*, failed: BaseException | None = None) -> None:
    if not _state.active:
        return
    if failed is None:
        trace_stage("decode_trace_complete")
    else:
        trace_stage(
            "decode_trace_failed",
            error_type=type(failed).__name__,
            error=str(failed),
        )
    _state.active = False


def synchronize(stage: str, device: torch.device | str | None = None, /, **fields: Any) -> None:
    """Synchronize the current accelerator after ``stage`` and bracket any HIP error."""

    if not _state.active:
        return
    start_fields = dict(fields)
    start_fields.update(_stream_fields(device))
    trace_stage(f"{stage}_sync_start", **start_fields)
    try:
        torch.cuda.synchronize(device)
    except BaseException as exc:
        trace_stage(
            f"{stage}_sync_failed",
            **fields,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    trace_stage(f"{stage}_sync_complete", **fields)


def _tensor_fields(tensor: torch.Tensor) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "strides": tuple(tensor.stride()),
        "contiguous": tensor.is_contiguous(),
        "storage_offset": tensor.storage_offset(),
        "is_view": tensor._base is not None,
    }
    try:
        fields["data_ptr"] = hex(tensor.data_ptr())
        fields["storage_ptr"] = hex(tensor.untyped_storage().data_ptr())
    except Exception as exc:
        fields["pointer_info"] = f"unavailable:{type(exc).__name__}"
    return fields


def trace_tensor(stage: str, tensor: torch.Tensor, *, include_minmax: bool = False) -> None:
    if not _state.active:
        return
    fields = _tensor_fields(tensor)
    if include_minmax and tensor.numel() > 0:
        if tensor.numel() > 4096:
            fields["minmax"] = "skipped_large_tensor"
        else:
            # ID tensors are tiny.  Copying after the preceding explicit synchronization
            # avoids launching advanced indexing merely to produce diagnostics.
            cpu = tensor.detach().reshape(-1).to("cpu")
            fields["min"] = cpu.min().item()
            fields["max"] = cpu.max().item()
    trace_stage(stage, **fields)


def snapshot_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Small diagnostic snapshot, detached and contiguous on the CPU."""

    return tensor.detach().to("cpu").contiguous()


def cache_hit_miss_counts(
    raw_expert_ids: torch.Tensor,
    slot_for_layer: torch.Tensor,
    num_experts: int,
) -> tuple[int, int, int]:
    """Return unique active, hit, and miss counts without device-side indexing."""

    raw = snapshot_cpu(raw_expert_ids).reshape(-1)
    if raw.dtype != torch.int32:
        raise TypeError(f"raw expert IDs must be torch.int32, got {raw.dtype}")
    if raw.numel() == 0:
        raise ValueError("raw expert IDs must not be empty during decode")
    lo, hi = int(raw.min()), int(raw.max())
    if lo < 0 or hi >= num_experts:
        raise ValueError(f"raw expert IDs out of range: min={lo}, max={hi}, experts={num_experts}")
    slots = snapshot_cpu(slot_for_layer).reshape(-1)
    unique = torch.unique(raw.to(torch.int64))
    resident = slots[unique]
    hits = int((resident >= 0).sum())
    active = int(unique.numel())
    return active, hits, active - hits


def _inspect_host_mapping(
    cache: Any,
    layer_id: int,
    extension: Any | None,
) -> _HostMappingStatus:
    """Prove that each current host bank has a HIP-visible device mapping."""

    residency = "unknown"
    try:
        residency = str(cache.layer_residency[layer_id])
    except (AttributeError, IndexError, TypeError):
        pass

    bank_names = tuple(getattr(cache, "bank_schema", ()))
    sources: list[tuple[str, torch.Tensor]] = []
    failed_bank = "none"
    reason = "ok"
    for name in bank_names:
        try:
            source = cache.bank_sources[name][layer_id]
        except (AttributeError, IndexError, KeyError, TypeError):
            failed_bank = str(name)
            reason = "host_bank_source_unavailable"
            break
        sources.append((str(name), source))

    first_host_ptr = "unavailable"
    if sources:
        try:
            first_host_ptr = hex(int(sources[0][1].data_ptr()))
        except Exception:
            reason = "host_pointer_unavailable"
            failed_bank = sources[0][0]

    if extension is None:
        return _HostMappingStatus(
            False,
            False,
            residency,
            len(bank_names),
            0,
            first_host_ptr,
            "unavailable",
            failed_bank if failed_bank != "none" else (sources[0][0] if sources else "none"),
            "pinned_extension_missing",
        )

    host_device_ptr = getattr(extension, "host_device_ptr", None)
    if not callable(host_device_ptr):
        return _HostMappingStatus(
            False,
            True,
            residency,
            len(bank_names),
            0,
            first_host_ptr,
            "unavailable",
            failed_bank if failed_bank != "none" else (sources[0][0] if sources else "none"),
            "host_device_ptr_unavailable",
        )

    if reason != "ok" or len(sources) != len(bank_names) or not sources:
        return _HostMappingStatus(
            False,
            True,
            residency,
            len(bank_names),
            0,
            first_host_ptr,
            "unavailable",
            failed_bank,
            reason if reason != "ok" else "host_bank_source_unavailable",
        )

    mapped_bank_count = 0
    first_device_ptr = "unavailable"
    for name, source in sources:
        if not isinstance(source, torch.Tensor) or source.device.type != "cpu":
            return _HostMappingStatus(
                False,
                True,
                residency,
                len(bank_names),
                mapped_bank_count,
                first_host_ptr,
                first_device_ptr,
                name,
                "host_bank_source_not_cpu",
            )
        try:
            mapped_ptr = int(host_device_ptr(int(source.data_ptr())))
        except Exception:
            return _HostMappingStatus(
                False,
                True,
                residency,
                len(bank_names),
                mapped_bank_count,
                first_host_ptr,
                first_device_ptr,
                name,
                "host_device_mapping_failed",
            )
        if mapped_ptr == 0:
            return _HostMappingStatus(
                False,
                True,
                residency,
                len(bank_names),
                mapped_bank_count,
                first_host_ptr,
                first_device_ptr,
                name,
                "host_device_pointer_zero",
            )
        mapped_bank_count += 1
        if first_device_ptr == "unavailable":
            first_device_ptr = hex(mapped_ptr)

    return _HostMappingStatus(
        True,
        True,
        residency,
        len(bank_names),
        mapped_bank_count,
        first_host_ptr,
        first_device_ptr,
        "none",
        "ok",
    )


def inspect_host_mapping(cache: Any, layer_id: int) -> _HostMappingStatus:
    """Inspect current bank mappings using the packaged pinned-memory extension."""

    from freetoken.kernel.pinned import _load_pinned_extension

    try:
        extension = _load_pinned_extension()
    except Exception:
        # Capability detection must fail closed: a broken/unloadable extension is not
        # evidence that a Windows host VA is safe for direct GPU dereference.
        return replace(
            _inspect_host_mapping(cache, layer_id, None),
            reason="pinned_extension_load_failed",
        )
    return _inspect_host_mapping(cache, layer_id, extension)


def preflight_windows_rocm_host_mapping(
    cache: Any,
    layer_id: int,
    *,
    allow_unmapped_safe_copy: bool = False,
) -> bool:
    """Fail before a traced Windows/ROCm decode copy can dereference host memory."""

    if not _state.active or sys.platform != "win32" or not getattr(torch.version, "hip", None):
        return False

    status = inspect_host_mapping(cache, layer_id)
    trace_stage(
        "host_mapping_preflight",
        platform="windows",
        rocm="true",
        pinned_extension_loaded=str(status.pinned_extension_loaded).lower(),
        host_bank_residency=status.host_bank_residency,
        bank_count=status.bank_count,
        mapped_bank_count=status.mapped_bank_count,
        host_ptr=status.host_ptr,
        device_ptr=status.device_ptr,
        failed_bank=status.failed_bank,
        safe_for_gpu_deref=str(status.safe_for_gpu_deref).lower(),
        reason=status.reason,
        layer=layer_id,
        safe_copy_selected=str(allow_unmapped_safe_copy).lower(),
    )
    if not status.safe_for_gpu_deref and not allow_unmapped_safe_copy:
        raise RuntimeError(
            "Windows ROCm offload decode trace refused copy_missing because the "
            "current host expert banks are not demonstrably pinned and HIP "
            f"device-mapped (reason={status.reason}, bank={status.failed_bank}). "
            "No GPU copy kernel was launched."
        )
    return True


def _validate_slot_ids(slot_ids: torch.Tensor, num_cache_slots: int) -> tuple[int, int]:
    """Pure validation used by the trace and CPU-only regression tests."""

    if slot_ids.dtype != torch.int32:
        raise TypeError(f"remapped slot IDs must be torch.int32, got {slot_ids.dtype}")
    if not slot_ids.is_contiguous():
        raise ValueError(f"remapped slot IDs must be contiguous, strides={slot_ids.stride()}")
    if slot_ids.numel() == 0:
        raise ValueError("remapped slot IDs must not be empty during decode")
    lo, hi = int(slot_ids.min()), int(slot_ids.max())
    if lo < 0 or hi >= num_cache_slots:
        raise ValueError(
            f"remapped slot IDs out of range: min={lo}, max={hi}, "
            f"num_cache_slots={num_cache_slots}"
        )
    return lo, hi


def validate_remapped_slots(
    raw_expert_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    id_of_slot: torch.Tensor,
    *,
    layer_id: int,
    num_experts: int,
    num_cache_slots: int,
) -> tuple[int, int, int]:
    """Validate slot geometry and the forward/reverse cache mapping before GEMM."""

    if raw_expert_ids.dtype != torch.int32:
        raise TypeError(f"raw expert IDs must be torch.int32, got {raw_expert_ids.dtype}")
    if slot_ids.dtype != torch.int32:
        raise TypeError(f"remapped slot IDs must be torch.int32, got {slot_ids.dtype}")
    if not slot_ids.is_contiguous():
        raise ValueError(f"remapped slot IDs must be contiguous, strides={slot_ids.stride()}")
    raw = snapshot_cpu(raw_expert_ids)
    slots = snapshot_cpu(slot_ids)
    owners = snapshot_cpu(id_of_slot).reshape(-1)
    if raw.shape != slots.shape:
        raise ValueError(f"raw/slot shape mismatch: raw={raw.shape}, slots={slots.shape}")
    lo, hi = _validate_slot_ids(slots, num_cache_slots)
    flat_slots = slots.reshape(-1).to(torch.int64)
    actual = owners[flat_slots]
    expected = layer_id * num_experts + raw.reshape(-1).to(actual.dtype)
    mismatched = actual != expected
    if bool(mismatched.any()):
        idx = int(torch.nonzero(mismatched, as_tuple=False)[0])
        raise ValueError(
            "cache slot is not populated by the routed expert: "
            f"route={idx}, raw_expert={int(raw.reshape(-1)[idx])}, "
            f"slot={int(flat_slots[idx])}, owner={int(actual[idx])}, "
            f"expected_owner={int(expected[idx])}"
        )
    return lo, hi, int(torch.unique(flat_slots).numel())


def _reset_for_tests() -> None:
    global _state
    _state = _TraceState()


__all__ = [
    "TRACE_ENV",
    "TRACE_PREFIX",
    "begin_first_decode",
    "cache_hit_miss_counts",
    "finish_trace",
    "inspect_host_mapping",
    "preflight_windows_rocm_host_mapping",
    "snapshot_cpu",
    "synchronize",
    "trace_active",
    "trace_enabled",
    "trace_stage",
    "trace_tensor",
    "validate_remapped_slots",
]
