from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.kernel import index as index_module


@pytest.mark.parametrize(
    "indices",
    [
        [2],
        [0, 3, 5],
        [2, 2, 1],
        [5, 0, 3, 1],
        [],
    ],
)
def test_gfx1201_unmasked_indexing_matches_index_select(monkeypatch, indices):
    monkeypatch.setattr(index_module, "_is_windows_rocm_gfx1201", lambda _device: True)
    weights = torch.arange(48, dtype=torch.bfloat16).view(6, 8)
    index = torch.tensor(indices, dtype=torch.int32)

    actual = index_module.indexing(weights, index)
    expected = torch.index_select(weights, 0, index)

    assert actual.dtype == weights.dtype
    assert actual.device == weights.device
    assert actual.shape == (len(indices), weights.shape[1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gfx1201_fallback_supports_noncontiguous_input(monkeypatch):
    monkeypatch.setattr(index_module, "_is_windows_rocm_gfx1201", lambda _device: True)
    weights = torch.arange(48, dtype=torch.bfloat16).view(6, 8)[:, ::2]
    index = torch.tensor([4, 1, 4], dtype=torch.int64)

    assert not weights.is_contiguous()
    torch.testing.assert_close(
        index_module.indexing(weights, index),
        torch.index_select(weights, 0, index),
        rtol=0,
        atol=0,
    )


def test_gfx1201_fallback_preserves_output_tensor(monkeypatch):
    monkeypatch.setattr(index_module, "_is_windows_rocm_gfx1201", lambda _device: True)
    weights = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    index = torch.tensor([5, 2, 0], dtype=torch.int32)
    output = torch.empty((3, 4), dtype=weights.dtype)

    result = index_module.indexing(weights, index, output=output)

    assert result is output
    torch.testing.assert_close(result, torch.index_select(weights, 0, index), rtol=0, atol=0)


def test_windows_rocm_gfx1201_detection(monkeypatch):
    monkeypatch.setattr(index_module.sys, "platform", "win32")
    monkeypatch.setattr(index_module.torch.version, "hip", "7.13")
    monkeypatch.setattr(
        index_module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName="gfx1201:sramecc-:xnack-"),
    )

    assert index_module._is_windows_rocm_gfx1201(torch.device("cuda:0")) is True


@pytest.mark.parametrize(
    ("platform", "hip", "arch"),
    [
        ("linux", "7.13", "gfx1201"),
        ("win32", None, "gfx1201"),
        ("win32", "7.13", "gfx1100"),
    ],
)
def test_gfx1201_detection_rejects_unaffected_runtimes(
    monkeypatch, platform, hip, arch
):
    get_properties = Mock(return_value=SimpleNamespace(gcnArchName=arch))
    monkeypatch.setattr(index_module.sys, "platform", platform)
    monkeypatch.setattr(index_module.torch.version, "hip", hip)
    monkeypatch.setattr(index_module.torch.cuda, "get_device_properties", get_properties)

    assert index_module._is_windows_rocm_gfx1201(torch.device("cuda:0")) is False
    if platform != "win32" or hip is None:
        get_properties.assert_not_called()


def test_non_gfx1201_preserves_optimized_kernel(monkeypatch):
    monkeypatch.setattr(index_module, "_is_windows_rocm_gfx1201", lambda _device: False)
    module = SimpleNamespace(launch=Mock())
    jit = Mock(return_value=module)
    monkeypatch.setattr(index_module, "_jit_index_module", jit)
    weights = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    index = torch.tensor([3, 1], dtype=torch.int32)

    result = index_module.indexing(weights, index)

    jit.assert_called_once_with(8, num_splits=1)
    module.launch.assert_called_once_with(weights, index, result, None)


def test_gfx1201_masked_path_preserves_optimized_kernel(monkeypatch):
    monkeypatch.setattr(index_module, "_is_windows_rocm_gfx1201", lambda _device: True)
    module = SimpleNamespace(launch=Mock())
    monkeypatch.setattr(index_module, "_jit_index_module", Mock(return_value=module))
    weights = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    index = torch.tensor([3, 1], dtype=torch.int32)

    result = index_module.indexing(weights, index, vocab_range=(1, 6))

    module.launch.assert_called_once_with(weights, index, result, (1, 6))
