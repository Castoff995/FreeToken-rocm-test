from __future__ import annotations

from unittest import mock

import pytest
import torch

from freetoken.kernel import pinned
from freetoken.moe.expert_banks import ExpertBanks
from freetoken.moe.host_banks import HostBank, HostResidency, PinPipeline


def _bank() -> HostBank:
    return HostBank((4, 16), torch.uint8)


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
