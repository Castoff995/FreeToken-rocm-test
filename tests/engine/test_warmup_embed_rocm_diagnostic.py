from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from freetoken.engine import engine


def _model_and_batch():
    weight = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    ids = torch.tensor([0, 3, 5], dtype=torch.int32)
    model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight))
    )
    batch = SimpleNamespace(input_ids=ids)
    return model, batch, weight, ids


def _active_patches(runtime):
    return (
        mock.patch.dict(
            "os.environ", {engine._ROCM_WARMUP_EMBED_DIAG_ENV: "1"}, clear=False
        ),
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned._load_hip_runtime", return_value=runtime
        ),
    )


def test_warmup_embed_diagnostic_is_noop_when_env_is_absent(monkeypatch):
    monkeypatch.delenv(engine._ROCM_WARMUP_EMBED_DIAG_ENV, raising=False)
    model, batch, _weight, _ids = _model_and_batch()
    with (
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch.object(
            engine.torch,
            "index_select",
            side_effect=AssertionError("disabled diagnostic ran"),
        ),
    ):
        assert engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        ) is None


def test_warmup_embed_diagnostic_is_noop_on_non_windows():
    model, batch, _weight, _ids = _model_and_batch()
    with (
        mock.patch.dict(
            "os.environ", {engine._ROCM_WARMUP_EMBED_DIAG_ENV: "1"}, clear=False
        ),
        mock.patch("freetoken.engine.engine.sys.platform", "linux"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch.object(
            engine.torch,
            "index_select",
            side_effect=AssertionError("non-Windows diagnostic ran"),
        ),
    ):
        assert engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        ) is None


def test_warmup_embed_diagnostic_is_noop_without_rocm():
    model, batch, _weight, _ids = _model_and_batch()
    with (
        mock.patch.dict(
            "os.environ", {engine._ROCM_WARMUP_EMBED_DIAG_ENV: "true"}, clear=False
        ),
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", None),
        mock.patch.object(
            engine.torch,
            "index_select",
            side_effect=AssertionError("non-ROCm diagnostic ran"),
        ),
    ):
        assert engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        ) is None


def test_matching_live_embedding_probes_pass_with_exact_inputs():
    model, batch, weight, ids = _model_and_batch()
    output = torch.index_select(weight, 0, ids)
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        mock.patch.object(engine.torch, "index_select", return_value=output) as torch_gather,
        mock.patch(
            "freetoken.kernel.index.indexing", return_value=output.clone()
        ) as ft_gather,
        mock.patch.object(engine.torch.cuda, "synchronize") as synchronize,
        mock.patch.object(engine.logger, "info_rank0") as log,
    ):
        state = engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        )

    assert torch_gather.call_args.args[0] is weight
    assert torch_gather.call_args.args[2] is ids
    assert ft_gather.call_args.args[0] is weight
    assert ft_gather.call_args.args[1] is ids
    assert synchronize.call_args_list == [
        mock.call(torch.device("cuda:0")),
        mock.call(torch.device("cuda:0")),
    ]
    assert runtime.hipGetLastError.call_count == 3
    assert state["weight_data_ptr"] == f"0x{weight.data_ptr():x}"
    assert state["ids_data_ptr"] == f"0x{ids.data_ptr():x}"
    assert state["outputs_equal"] is True
    assert state["result"] == "PASS"
    assert any(call.args[0].endswith("PASS") for call in log.call_args_list)


def test_initial_sticky_hip_error_stops_before_both_probes():
    model, batch, _weight, _ids = _model_and_batch()
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=17))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch, "index_select") as torch_gather,
        mock.patch("freetoken.kernel.index.indexing") as ft_gather,
        pytest.raises(RuntimeError, match="hip_last_error_before_probe=FAIL"),
    ):
        engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        )

    runtime.hipGetLastError.assert_called_once_with()
    torch_gather.assert_not_called()
    ft_gather.assert_not_called()


def test_pytorch_stage_failure_stops_before_freetoken_indexing():
    model, batch, _weight, _ids = _model_and_batch()
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=0))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(
            engine.torch, "index_select", side_effect=RuntimeError("torch gather failed")
        ),
        mock.patch("freetoken.kernel.index.indexing") as ft_gather,
        pytest.raises(RuntimeError, match="torch_index_select=FAIL"),
    ):
        engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        )

    ft_gather.assert_not_called()


def test_freetoken_stage_failure_occurs_after_pytorch_success():
    model, batch, weight, ids = _model_and_batch()
    output = torch.index_select(weight, 0, ids)
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch, "index_select", return_value=output) as torch_gather,
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch(
            "freetoken.kernel.index.indexing",
            side_effect=RuntimeError("FreeToken launch failed"),
        ),
        pytest.raises(RuntimeError, match="freetoken_indexing=FAIL"),
    ):
        engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        )

    torch_gather.assert_called_once()


def test_output_mismatch_is_reported_as_failure():
    model, batch, weight, ids = _model_and_batch()
    torch_output = torch.index_select(weight, 0, ids)
    ft_output = torch_output.clone()
    ft_output[0, 0] += 1
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch, "index_select", return_value=torch_output),
        mock.patch("freetoken.kernel.index.indexing", return_value=ft_output),
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine.logger, "warning_rank0") as warning,
        pytest.raises(RuntimeError, match="output_compare=FAIL"),
    ):
        engine._run_windows_rocm_warmup_embed_diagnostic(
            model, batch, torch.device("cuda:0"), 3
        )

    assert any("max_abs_diff=" in call.args[0] for call in warning.call_args_list)


def test_missing_embedding_weight_logs_and_skips_cleanly():
    batch = SimpleNamespace(input_ids=torch.tensor([0], dtype=torch.int32))
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=0))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.logger, "warning_rank0") as warning,
    ):
        state = engine._run_windows_rocm_warmup_embed_diagnostic(
            SimpleNamespace(), batch, torch.device("cuda:0"), 1
        )

    assert state == {"skipped": "embedding_weight_unavailable"}
    assert "action=skip" in warning.call_args.args[0]
    runtime.hipGetLastError.assert_not_called()


class _FakeEvent:
    def record(self, _stream):
        pass

    def elapsed_time(self, _other):
        return 1.0


def test_warmup_prefill_invokes_embedding_diagnostic_only_once():
    page_table = torch.zeros((1, 256), dtype=torch.int32)
    fake_engine = SimpleNamespace(
        max_seq_len=256,
        page_table=page_table,
        dummy_req=SimpleNamespace(table_idx=0),
        stream=object(),
        device=torch.device("cpu"),
        attn_backend=SimpleNamespace(prepare_metadata=mock.Mock()),
        ctx=SimpleNamespace(forward_batch=lambda _batch: nullcontext()),
        model=SimpleNamespace(forward=mock.Mock()),
        moe_offload_cache=None,
    )
    with (
        mock.patch.object(engine.torch.cuda, "Event", side_effect=[_FakeEvent(), _FakeEvent()]),
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(
            engine, "_run_windows_rocm_warmup_embed_diagnostic"
        ) as diagnostic,
        mock.patch.object(engine.logger, "info_rank0"),
    ):
        engine.Engine._warmup_prefill(fake_engine)

    diagnostic.assert_called_once()
    assert diagnostic.call_args.args[3] == 80
    assert fake_engine.attn_backend.prepare_metadata.call_count == 2
    assert fake_engine.model.forward.call_count == 2
