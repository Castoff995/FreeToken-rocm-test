from types import SimpleNamespace

from freetoken.utils import arch


def test_triton_pdl_launch_kwargs_omits_nvidia_option_on_rocm(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "version", SimpleNamespace(hip="7.13.99004"))
    assert arch.triton_pdl_launch_kwargs(False) == {}
    assert arch.triton_pdl_launch_kwargs(True) == {}


def test_triton_pdl_launch_kwargs_preserves_nvidia_behavior(monkeypatch):
    import torch

    monkeypatch.setattr(torch, "version", SimpleNamespace(hip=None))
    assert arch.triton_pdl_launch_kwargs(False) == {"launch_pdl": False}
    assert arch.triton_pdl_launch_kwargs(True) == {"launch_pdl": True}
