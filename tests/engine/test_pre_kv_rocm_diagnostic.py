from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from freetoken.engine import engine


@pytest.mark.parametrize(
    ("platform", "hip_version"),
    [("linux", "test-rocm"), ("win32", None)],
)
def test_pre_kv_diagnostic_is_inert_outside_windows_rocm(platform, hip_version):
    with (
        mock.patch("freetoken.engine.engine.sys.platform", platform),
        mock.patch.object(torch.version, "hip", hip_version),
        mock.patch.object(
            engine.torch.cuda,
            "is_available",
            side_effect=AssertionError("inactive diagnostic touched CUDA"),
        ),
    ):
        assert engine._run_windows_rocm_pre_kv_diagnostic(
            torch.device("cuda:0")
        ) is None


def test_pre_kv_diagnostic_returns_structured_state_and_logs_stages():
    device = torch.device("cuda:0")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=7))
    allocations = [object(), object()]
    with (
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned._load_hip_runtime", return_value=runtime
        ),
        mock.patch.object(engine.torch.cuda, "is_available", return_value=True),
        mock.patch.object(engine.torch.cuda, "current_device", side_effect=[0, 0]),
        mock.patch.object(
            engine.torch.cuda, "get_device_name", return_value="Test GPU"
        ),
        mock.patch.object(
            engine.torch.cuda,
            "mem_get_info",
            return_value=(3 << 30, 16 << 30),
        ),
        mock.patch.object(engine.torch, "empty", side_effect=allocations) as empty,
        mock.patch.object(engine.torch.cuda, "synchronize") as synchronize,
        mock.patch.object(engine.torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(engine.logger, "info_rank0") as log,
    ):
        state = engine._run_windows_rocm_pre_kv_diagnostic(device)

    assert state == {
        "cuda_available": True,
        "device": "cuda:0",
        "current_device_before": 0,
        "device_name": "Test GPU",
        "hip_last_error_before_clear": 7,
        "memory_free_bytes": 3 << 30,
        "memory_total_bytes": 16 << 30,
        "probe_1byte": "ok",
        "probe_1MiB": "ok",
        "empty_cache": "ok",
        "current_device_after": 0,
    }
    runtime.hipGetLastError.assert_called_once_with()
    assert empty.call_args_list == [
        mock.call(1, dtype=torch.uint8, device=device),
        mock.call(1 << 20, dtype=torch.uint8, device=device),
    ]
    assert synchronize.call_args_list == [mock.call(device), mock.call(device)]
    empty_cache.assert_called_once_with()
    messages = [call.args[0] for call in log.call_args_list]
    assert any("hip_last_error_before_clear=7" in message for message in messages)
    assert any("probe_1byte=ok" in message for message in messages)
    assert any("probe_1MiB=ok" in message for message in messages)
    assert any("current_device_after_probes=0" in message for message in messages)


def test_failed_first_probe_stops_before_later_probe_and_real_allocation():
    device = torch.device("cuda:0")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=0))
    with (
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned._load_hip_runtime", return_value=runtime
        ),
        mock.patch.object(engine.torch.cuda, "is_available", return_value=True),
        mock.patch.object(engine.torch.cuda, "current_device", return_value=0) as current,
        mock.patch.object(
            engine.torch.cuda, "get_device_name", return_value="Test GPU"
        ),
        mock.patch.object(
            engine.torch.cuda, "mem_get_info", return_value=(1024, 2048)
        ),
        mock.patch.object(
            engine.torch, "empty", side_effect=RuntimeError("probe allocation failed")
        ) as empty,
        mock.patch.object(engine.torch.cuda, "synchronize") as synchronize,
        mock.patch.object(engine.torch.cuda, "empty_cache") as empty_cache,
        mock.patch.object(engine.logger, "warning_rank0") as warning,
        pytest.raises(RuntimeError, match="probe_1byte=FAIL"),
    ):
        engine._run_windows_rocm_pre_kv_diagnostic(device)

    empty.assert_called_once_with(1, dtype=torch.uint8, device=device)
    synchronize.assert_not_called()
    empty_cache.assert_not_called()
    current.assert_called_once_with()
    assert "probe allocation failed" in warning.call_args.args[0]
