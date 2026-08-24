from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest
import torch

from freetoken.kernel import pinned
from freetoken.moe.expert_banks import ExpertBanks
from freetoken.moe.host_banks import (
    _GIB,
    _PIN_BUDGET_ENV,
    HostBank,
    HostResidency,
    PinPipeline,
)
from freetoken.moe.offload_cache import OffloadMoeCache


def _bank() -> HostBank:
    return HostBank((4, 16), torch.uint8)


def _budget_for_bytes(nbytes: int) -> str:
    return format(Decimal(nbytes) / Decimal(_GIB), "f")


@pytest.fixture(autouse=True)
def _clear_pin_budget(monkeypatch):
    monkeypatch.delenv(_PIN_BUDGET_ENV, raising=False)


def test_windows_rocm_pin_budget_env_absent_preserves_registration_behavior():
    first, second = _bank(), _bank()
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch("freetoken.kernel.pinned.host_register") as register,
        mock.patch("freetoken.moe.host_banks.logger.info") as info,
    ):
        with PinPipeline() as pins:
            pins.submit(first)
            pins.submit(second)

    assert register.call_count == 2
    assert first.residency == second.residency == HostResidency.PINNED
    assert not any("host pin budget" in call.args[0] for call in info.call_args_list)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("12", 12 * _GIB), ("12.5", 25 * _GIB // 2)],
)
def test_windows_rocm_pin_budget_accepts_integer_and_decimal(monkeypatch, raw, expected):
    monkeypatch.setenv(_PIN_BUDGET_ENV, raw)
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        PinPipeline() as pins,
    ):
        assert pins._budget_bytes == expected


@pytest.mark.parametrize("raw", ["", "bad", "0", "-1", "NaN", "Inf", "-Infinity"])
def test_windows_rocm_pin_budget_rejects_invalid_values(monkeypatch, raw):
    monkeypatch.setenv(_PIN_BUDGET_ENV, raw)
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        pytest.raises(ValueError, match=f"{_PIN_BUDGET_ENV}.*greater than zero"),
    ):
        PinPipeline()


def test_windows_rocm_pin_budget_larger_than_all_banks_pins_everything(monkeypatch):
    banks = {"gate_up": _bank(), "down": _bank()}
    monkeypatch.setenv(_PIN_BUDGET_ENV, "1")
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch("freetoken.kernel.pinned.host_register") as register,
    ):
        with PinPipeline() as pins:
            pins(0, banks)

    assert register.call_count == 2
    assert all(bank.residency == HostResidency.PINNED for bank in banks.values())
    assert pins._mapped_layers == 1
    assert pins._budget_skipped_banks == 0


def test_windows_rocm_pin_budget_exact_fit_is_allowed(monkeypatch):
    banks = {"gate_up": _bank(), "down": _bank()}
    exact_bytes = sum(bank.registration_nbytes for bank in banks.values())
    monkeypatch.setenv(_PIN_BUDGET_ENV, _budget_for_bytes(exact_bytes))
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch("freetoken.kernel.pinned.host_register") as register,
    ):
        with PinPipeline() as pins:
            pins(0, banks)

    assert register.call_count == 2
    assert pins._successfully_pinned_bytes == exact_bytes
    assert pins._budget_skipped_bytes == 0


def test_windows_rocm_pin_budget_skips_whole_layer_and_selects_safe_copy(monkeypatch):
    layer_0 = {"gate_up": _bank(), "down": _bank()}
    layer_1 = {"gate_up": _bank(), "down": _bank()}
    layer_bytes = sum(bank.registration_nbytes for bank in layer_0.values())
    monkeypatch.setenv(_PIN_BUDGET_ENV, _budget_for_bytes(layer_bytes))
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch("freetoken.moe.offload_cache.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch("freetoken.kernel.pinned.host_register") as register,
        mock.patch(
            "freetoken.kernel.pinned._clear_recoverable_hip_host_register_error"
        ) as cleanup,
        mock.patch("freetoken.moe.host_banks.logger.info") as info,
    ):
        with PinPipeline() as pins:
            pins(0, layer_0)
            pins(1, layer_1)

        bundle = ExpertBanks(
            "bf16",
            {
                "gate_up": [layer_0["gate_up"].tensor, layer_1["gate_up"].tensor],
                "down": [layer_0["down"].tensor, layer_1["down"].tensor],
            },
        )
        cache = OffloadMoeCache(
            num_layers=2,
            num_experts=4,
            cache_size=6,
            device=torch.device("cpu"),
        )
        cache.set_bank_sources(bundle.sources, layer_residency=bundle.layer_residency)

        assert bundle.layer_residency == ["pinned", "pageable"]
        assert cache._copy_fused_ok_by_layer[1] is False
        assert cache.should_use_safe_offload_copy(1) is True

    assert register.call_count == 2
    assert cleanup.call_count == 0
    assert all(bank.residency == HostResidency.PINNED for bank in layer_0.values())
    assert all(bank.residency == HostResidency.PAGEABLE for bank in layer_1.values())
    assert all(bank._pin_error is None for bank in layer_1.values())
    assert pins._attempted_pin_bytes == layer_bytes
    assert pins._successfully_pinned_bytes == layer_bytes
    assert pins._budget_skipped_bytes == layer_bytes
    assert pins._mapped_count == 2
    assert pins._pageable_count == 2
    assert pins._budget_skipped_banks == 2
    assert pins._budget_skipped_layer_ids == {1}
    assert pins._thread.is_alive() is False
    summary = next(
        call.args[0] for call in info.call_args_list if "host pin summary" in call.args[0]
    )
    assert "mapped_banks=2" in summary
    assert "pageable_banks=2" in summary
    assert "budget_skipped_banks=2" in summary
    assert "budget_skipped_layers=1" in summary
    assert "first_registration_failure=none" in summary


def test_non_windows_ignores_pin_budget_and_preserves_registration(monkeypatch):
    bank = _bank()
    monkeypatch.setenv(_PIN_BUDGET_ENV, "not-a-number")
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "linux"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch("freetoken.kernel.pinned.host_register") as register,
    ):
        with PinPipeline() as pins:
            pins.submit(bank)

    register.assert_called_once()
    assert pins._budget_bytes is None
    assert bank.residency == HostResidency.PINNED


def test_windows_rocm_registration_failure_under_budget_keeps_cleanup_fallback(monkeypatch):
    first, second = _bank(), _bank()
    runtime = mock.Mock()
    runtime.hipGetLastError.return_value = 17
    registration_failure = pinned._HipHostRegisterError(
        runtime, first.registration_nbytes, 17
    )
    monkeypatch.setenv(_PIN_BUDGET_ENV, "1")
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned.host_register",
            side_effect=[registration_failure, None],
        ) as register,
        mock.patch("freetoken.moe.host_banks.logger.warning") as warning,
    ):
        with PinPipeline() as pins:
            pins(0, {"gate_up": first, "down": second})

    assert register.call_count == 2
    runtime.hipGetLastError.assert_called_once_with()
    assert first.residency == HostResidency.PAGEABLE
    assert second.residency == HostResidency.PINNED
    assert pins._attempted_pin_bytes == 2 * first.registration_nbytes
    assert pins._successfully_pinned_bytes == second.registration_nbytes
    assert pins._budget_skipped_bytes == 0
    assert pins._budget_skipped_banks == 0
    assert pins._mapped_layers == 0
    warning.assert_called_once()
    assert "first_registration_failure=" in warning.call_args.args[0]
    assert "hipError 17" in warning.call_args.args[0]


def test_windows_rocm_pin_pipeline_continues_after_registration_failure():
    gate_0, down_0, gate_1, down_1 = (_bank() for _ in range(4))
    runtime = mock.Mock()
    runtime.hipGetLastError.return_value = 1
    registration_failure = pinned._HipHostRegisterError(runtime, 128 << 20, 1)
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned.host_register",
            side_effect=[None, None, registration_failure, None],
        ) as register,
        mock.patch("freetoken.moe.host_banks.logger.warning") as warning,
    ):
        with PinPipeline() as pins:
            for bank in (gate_0, down_0, gate_1, down_1):
                pins.submit(bank)

    assert register.call_count == 4
    assert [bank.residency for bank in (gate_0, down_0, gate_1, down_1)] == [
        HostResidency.PINNED,
        HostResidency.PINNED,
        HostResidency.PAGEABLE,
        HostResidency.PINNED,
    ]
    warning.assert_called_once()
    assert "mapped_banks=3" in warning.call_args.args[0]
    assert "pageable_banks=1" in warning.call_args.args[0]
    assert "hipError 1" in warning.call_args.args[0]
    runtime.hipGetLastError.assert_called_once_with()

    bundle = ExpertBanks(
        "bf16",
        {
            "gate_up": [gate_0.tensor, gate_1.tensor],
            "down": [down_0.tensor, down_1.tensor],
        },
    )
    assert bundle.layer_residency == [
        HostResidency.PINNED.value,
        HostResidency.PAGEABLE.value,
    ]


def test_non_windows_pin_pipeline_preserves_fail_fast_behavior():
    first, second = _bank(), _bank()
    runtime = mock.Mock()
    registration_failure = pinned._HipHostRegisterError(runtime, 4096, 17)
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "linux"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned.host_register",
            side_effect=registration_failure,
        ) as register,
        pytest.raises(RuntimeError, match="host registration failed"),
    ):
        with PinPipeline() as pins:
            pins.submit(first)
            pins.submit(second)

    register.assert_called_once()
    runtime.hipGetLastError.assert_not_called()
    assert first.residency == HostResidency.PAGEABLE
    assert second.residency == HostResidency.PAGEABLE


def test_windows_rocm_pin_pipeline_does_not_continue_if_error_clear_fails():
    bank = _bank()
    runtime = mock.Mock()
    runtime.hipGetLastError.side_effect = RuntimeError("clear failed")
    registration_failure = pinned._HipHostRegisterError(runtime, 4096, 17)
    with (
        mock.patch("freetoken.moe.host_banks.sys.platform", "win32"),
        mock.patch.object(torch.version, "hip", "test-rocm"),
        mock.patch(
            "freetoken.kernel.pinned.host_register",
            side_effect=registration_failure,
        ),
        pytest.raises(RuntimeError, match="clear failed"),
    ):
        with PinPipeline() as pins:
            pins.submit(bank)

    runtime.hipGetLastError.assert_called_once_with()
