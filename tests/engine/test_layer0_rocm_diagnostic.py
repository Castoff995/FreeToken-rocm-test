from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from freetoken.engine import engine


def _live_stack():
    ids = torch.tensor([0, 1, 2], dtype=torch.int32)
    hidden = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    normed = hidden + 1
    qkvz = torch.arange(30, dtype=torch.bfloat16).view(3, 10)
    ba = torch.arange(6, dtype=torch.bfloat16).view(3, 2)
    embedding = SimpleNamespace(forward=mock.Mock(return_value=hidden))
    norm = SimpleNamespace(forward=mock.Mock(return_value=normed))
    qkvz_op = SimpleNamespace(
        forward=mock.Mock(return_value=qkvz),
        weight=torch.empty((10, 4), dtype=torch.uint8),
        weight_scale=torch.ones(10, dtype=torch.float32),
        input_scale=torch.ones(1, dtype=torch.float32),
    )
    ba_op = SimpleNamespace(
        forward=mock.Mock(return_value=ba),
        weight=torch.empty((2, 4), dtype=torch.bfloat16),
    )
    gdn = SimpleNamespace(
        _pertensor_fp8=True,
        _block_fp8=False,
        _fp8=True,
        in_proj_qkvz=qkvz_op,
        in_proj_ba=ba_op,
    )
    layer0 = SimpleNamespace(input_layernorm=norm, linear_attn=gdn)
    model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=embedding,
            layers=SimpleNamespace(op_list=[layer0]),
        )
    )
    batch = SimpleNamespace(input_ids=ids)
    return SimpleNamespace(
        model=model,
        batch=batch,
        ids=ids,
        hidden=hidden,
        normed=normed,
        qkvz=qkvz,
        ba=ba,
        embedding=embedding,
        norm=norm,
        qkvz_op=qkvz_op,
        ba_op=ba_op,
    )


def _active_patches(runtime):
    return (
        mock.patch.dict(
            "os.environ", {engine._ROCM_LAYER0_DIAG_ENV: "1"}, clear=False
        ),
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned._load_hip_runtime", return_value=runtime
        ),
    )


def _assert_later_stages_not_called(stack, *, norm=False, qkvz=False, ba=False):
    (stack.norm.forward.assert_called_once() if norm else stack.norm.forward.assert_not_called())
    (
        stack.qkvz_op.forward.assert_called_once()
        if qkvz
        else stack.qkvz_op.forward.assert_not_called()
    )
    (stack.ba_op.forward.assert_called_once() if ba else stack.ba_op.forward.assert_not_called())


def test_layer0_diagnostic_is_noop_when_env_absent(monkeypatch):
    monkeypatch.delenv(engine._ROCM_LAYER0_DIAG_ENV, raising=False)
    stack = _live_stack()
    with (
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
    ):
        assert engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        ) is None
    stack.embedding.forward.assert_not_called()


def test_layer0_diagnostic_is_noop_on_non_windows():
    stack = _live_stack()
    with (
        mock.patch.dict("os.environ", {engine._ROCM_LAYER0_DIAG_ENV: "yes"}),
        mock.patch("freetoken.engine.engine.sys.platform", "linux"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
    ):
        assert engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        ) is None
    stack.embedding.forward.assert_not_called()


def test_layer0_diagnostic_is_noop_without_rocm():
    stack = _live_stack()
    with (
        mock.patch.dict("os.environ", {engine._ROCM_LAYER0_DIAG_ENV: "on"}),
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", None),
    ):
        assert engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        ) is None
    stack.embedding.forward.assert_not_called()


def test_missing_layer0_structure_skips_cleanly():
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=0))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.logger, "warning_rank0") as warning,
    ):
        state = engine._run_windows_rocm_layer0_diagnostic(
            SimpleNamespace(),
            SimpleNamespace(input_ids=torch.tensor([0], dtype=torch.int32)),
            torch.device("cuda:0"),
            1,
        )
    assert state == {"skipped": "compatible_layer0_unavailable"}
    assert "action=skip" in warning.call_args.args[0]
    runtime.hipGetLastError.assert_not_called()


def test_initial_hip_error_stops_before_embedding_input():
    stack = _live_stack()
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=17))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        pytest.raises(RuntimeError, match="hip_last_error_before_probe=FAIL"),
    ):
        engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    stack.embedding.forward.assert_not_called()
    _assert_later_stages_not_called(stack)


def test_embedding_input_failure_stops_before_layer0():
    stack = _live_stack()
    stack.embedding.forward.side_effect = RuntimeError("embedding failed")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(return_value=0))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        pytest.raises(RuntimeError, match="embedding_input=FAIL"),
    ):
        engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    _assert_later_stages_not_called(stack)


def test_input_layernorm_failure_classifies_l0_a():
    stack = _live_stack()
    stack.norm.forward.side_effect = RuntimeError("norm failed")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine.logger, "warning_rank0") as warning,
        pytest.raises(RuntimeError, match="classification=L0-A"),
    ):
        engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    _assert_later_stages_not_called(stack, norm=True)
    assert any("CLASSIFICATION=L0-A" in call.args[0] for call in warning.call_args_list)


def test_qkvz_failure_classifies_l0_b_and_skips_ba():
    stack = _live_stack()
    stack.qkvz_op.forward.side_effect = RuntimeError("qkvz failed")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine.logger, "warning_rank0") as warning,
        pytest.raises(RuntimeError, match="classification=L0-B"),
    ):
        engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    _assert_later_stages_not_called(stack, norm=True, qkvz=True)
    assert any("CLASSIFICATION=L0-B" in call.args[0] for call in warning.call_args_list)


def test_ba_failure_classifies_l0_c_after_norm_and_qkvz():
    stack = _live_stack()
    stack.ba_op.forward.side_effect = RuntimeError("ba failed")
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0, 0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine.logger, "warning_rank0") as warning,
        pytest.raises(RuntimeError, match="classification=L0-C"),
    ):
        engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    _assert_later_stages_not_called(stack, norm=True, qkvz=True, ba=True)
    assert any("CLASSIFICATION=L0-C" in call.args[0] for call in warning.call_args_list)


def test_all_layer0_stages_pass_with_exact_live_operators_and_checkpoints():
    stack = _live_stack()
    runtime = SimpleNamespace(hipGetLastError=mock.Mock(side_effect=[0, 0, 0, 0, 0]))
    patches = _active_patches(runtime)
    with (
        patches[0], patches[1], patches[2], patches[3],
        mock.patch.object(engine.torch.cuda, "synchronize") as synchronize,
        mock.patch.object(engine.logger, "info_rank0") as log,
    ):
        state = engine._run_windows_rocm_layer0_diagnostic(
            stack.model, stack.batch, torch.device("cuda:0"), 3
        )
    assert stack.embedding.forward.call_args.args[0] is stack.ids
    assert stack.norm.forward.call_args.args[0] is stack.hidden
    assert stack.qkvz_op.forward.call_args.args[0] is stack.normed
    assert stack.ba_op.forward.call_args.args[0] is stack.normed
    assert synchronize.call_args_list == [mock.call(torch.device("cuda:0"))] * 4
    assert runtime.hipGetLastError.call_count == 5
    assert state["classification"] == "L0-D"
    assert state["result"] == "PASS"
    assert state["_pertensor_fp8"] is True
    assert state["_block_fp8"] is False
    assert state["_fp8"] is True
    assert any("CLASSIFICATION=L0-D" in call.args[0] for call in log.call_args_list)


class _FakeEvent:
    def record(self, _stream):
        pass

    def elapsed_time(self, _other):
        return 1.0


def _warmup_engine():
    return SimpleNamespace(
        max_seq_len=256,
        page_table=torch.zeros((1, 256), dtype=torch.int32),
        dummy_req=SimpleNamespace(table_idx=0),
        stream=object(),
        device=torch.device("cpu"),
        attn_backend=SimpleNamespace(prepare_metadata=mock.Mock()),
        ctx=SimpleNamespace(forward_batch=lambda _batch: nullcontext()),
        model=SimpleNamespace(forward=mock.Mock()),
        moe_offload_cache=None,
    )


def test_warmup_prefill_invokes_layer0_diagnostic_only_once():
    fake_engine = _warmup_engine()
    with (
        mock.patch.object(engine.torch.cuda, "Event", side_effect=[_FakeEvent(), _FakeEvent()]),
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine, "_run_windows_rocm_warmup_embed_diagnostic"),
        mock.patch.object(engine, "_run_windows_rocm_layer0_diagnostic") as diagnostic,
        mock.patch.object(engine.logger, "info_rank0"),
    ):
        engine.Engine._warmup_prefill(fake_engine)
    diagnostic.assert_called_once()
    assert diagnostic.call_args.args[3] == 80
    assert fake_engine.model.forward.call_count == 2


def test_warmup_prefill_normal_forwards_unchanged_when_layer0_gate_is_off(monkeypatch):
    monkeypatch.delenv(engine._ROCM_LAYER0_DIAG_ENV, raising=False)
    fake_engine = _warmup_engine()
    with (
        mock.patch.object(engine.torch.cuda, "Event", side_effect=[_FakeEvent(), _FakeEvent()]),
        mock.patch.object(engine.torch.cuda, "synchronize"),
        mock.patch.object(engine, "_run_windows_rocm_warmup_embed_diagnostic"),
        mock.patch("freetoken.engine.engine.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch.object(engine.logger, "info_rank0"),
    ):
        engine.Engine._warmup_prefill(fake_engine)
    assert fake_engine.attn_backend.prepare_metadata.call_count == 2
    assert fake_engine.model.forward.call_count == 2
