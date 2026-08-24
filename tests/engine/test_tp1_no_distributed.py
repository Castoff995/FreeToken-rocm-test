from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.engine.engine import Engine
from freetoken.scheduler.io import SchedulerIOMixin


def _communication_config(*, size: int, use_pynccl: bool = False):
    return SimpleNamespace(
        tp_info=DistributedInfo(rank=0, size=size),
        use_pynccl=use_pynccl,
        distributed_timeout=17.0,
        distributed_addr="tcp://127.0.0.1:2333",
    )


def test_tp1_skips_unavailable_torch_distributed(monkeypatch):
    from freetoken.engine import engine as engine_module

    unavailable_dist = SimpleNamespace(is_available=Mock(return_value=False))
    monkeypatch.setattr(engine_module.torch, "distributed", unavailable_dist)

    instance = object.__new__(Engine)
    assert instance._init_communication(_communication_config(size=1)) is None
    unavailable_dist.is_available.assert_not_called()


def test_tp1_skips_process_group_when_distributed_is_available(monkeypatch):
    from freetoken.engine import engine as engine_module

    available_dist = SimpleNamespace(
        is_available=Mock(return_value=True),
        init_process_group=Mock(side_effect=AssertionError("must not initialize TP=1")),
    )
    monkeypatch.setattr(engine_module.torch, "distributed", available_dist)

    instance = object.__new__(Engine)
    assert instance._init_communication(_communication_config(size=1)) is None
    available_dist.is_available.assert_not_called()
    available_dist.init_process_group.assert_not_called()


def test_tp_gt_1_fails_early_without_torch_distributed(monkeypatch):
    from freetoken.engine import engine as engine_module

    unavailable_dist = SimpleNamespace(is_available=Mock(return_value=False))
    monkeypatch.setattr(engine_module.torch, "distributed", unavailable_dist)

    instance = object.__new__(Engine)
    with pytest.raises(RuntimeError, match="tensor parallel size > 1 requires PyTorch distributed"):
        instance._init_communication(_communication_config(size=2))


def test_tp_gt_1_keeps_existing_process_group_path(monkeypatch):
    from freetoken.engine import engine as engine_module

    cpu_group = object()
    available_dist = SimpleNamespace(
        is_available=Mock(return_value=True),
        init_process_group=Mock(),
        new_group=Mock(return_value=cpu_group),
    )
    monkeypatch.setattr(engine_module.torch, "distributed", available_dist)

    instance = object.__new__(Engine)
    result = instance._init_communication(_communication_config(size=2))

    assert result is cpu_group
    available_dist.init_process_group.assert_called_once_with(
        backend="nccl",
        rank=0,
        world_size=2,
        timeout=engine_module.timedelta(seconds=17.0),
        init_method="tcp://127.0.0.1:2333",
    )
    available_dist.new_group.assert_called_once_with(backend="gloo")


def test_tp_gt_1_keeps_existing_pynccl_process_group_path(monkeypatch):
    from freetoken.engine import engine as engine_module

    cpu_group = object()
    available_dist = SimpleNamespace(
        is_available=Mock(return_value=True),
        init_process_group=Mock(),
        group=SimpleNamespace(WORLD=cpu_group),
    )
    enable_pynccl = Mock()
    monkeypatch.setattr(engine_module.torch, "distributed", available_dist)
    monkeypatch.setattr(engine_module, "enable_pynccl_distributed", enable_pynccl)

    config = _communication_config(size=2, use_pynccl=True)
    config.max_forward_len = 64
    config.model_config = SimpleNamespace(hidden_size=128)
    instance = object.__new__(Engine)
    instance.dtype = torch.bfloat16

    result = instance._init_communication(config)

    assert result is cpu_group
    available_dist.init_process_group.assert_called_once_with(
        backend="gloo",
        rank=0,
        world_size=2,
        timeout=engine_module.timedelta(seconds=17.0),
        init_method="tcp://127.0.0.1:2333",
    )
    enable_pynccl.assert_called_once_with(
        config.tp_info,
        cpu_group,
        64 * 128 * torch.bfloat16.itemsize,
    )


def test_tp1_memory_sync_is_local(monkeypatch):
    from freetoken.engine import engine as engine_module

    all_reduce = Mock(side_effect=AssertionError("must not reduce TP=1 memory"))
    monkeypatch.setattr(
        engine_module.torch,
        "distributed",
        SimpleNamespace(all_reduce=all_reduce),
    )
    monkeypatch.setattr(engine_module.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(engine_module.torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(engine_module.torch.cuda, "reset_peak_memory_stats", Mock())
    monkeypatch.setattr(engine_module, "get_free_memory", lambda _device: 123456)

    instance = object.__new__(Engine)
    instance.config = SimpleNamespace(tp_info=DistributedInfo(rank=0, size=1))
    instance.device = torch.device("cpu")

    assert instance._sync_get_memory() == (123456, 123456)
    all_reduce.assert_not_called()


def test_tp1_collectives_are_identity(monkeypatch):
    from freetoken.distributed import impl, info

    plugin = SimpleNamespace(
        all_reduce=Mock(side_effect=AssertionError("must not reduce TP=1")),
        all_gather=Mock(side_effect=AssertionError("must not gather TP=1")),
    )
    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(rank=0, size=1))
    monkeypatch.setattr(impl.DistributedCommunicator, "plugins", [plugin])

    communicator = impl.DistributedCommunicator()
    value = torch.tensor([1.0, 2.0])

    assert communicator.all_reduce(value) is value
    assert communicator.all_gather(value) is value
    plugin.all_reduce.assert_not_called()
    plugin.all_gather.assert_not_called()


def test_tp1_scheduler_barrier_is_noop():
    scheduler_io = object.__new__(SchedulerIOMixin)
    scheduler_io.tp_cpu_group = None
    scheduler_io.sync_all_ranks()


def test_tp_gt_1_scheduler_barrier_is_unchanged():
    wait = Mock()
    group = SimpleNamespace(barrier=Mock(return_value=SimpleNamespace(wait=wait)))
    scheduler_io = object.__new__(SchedulerIOMixin)
    scheduler_io.tp_cpu_group = group

    scheduler_io.sync_all_ranks()

    group.barrier.assert_called_once_with()
    wait.assert_called_once_with()


def test_tp1_shutdown_skips_process_group_destroy(monkeypatch):
    from freetoken.engine import engine as engine_module

    destroy_process_group = Mock(side_effect=AssertionError("no TP=1 group exists"))
    destroy_plugins = Mock()
    monkeypatch.setattr(
        engine_module.torch,
        "distributed",
        SimpleNamespace(destroy_process_group=destroy_process_group),
    )
    monkeypatch.setattr(engine_module, "destroy_distributed", destroy_plugins)

    instance = object.__new__(Engine)
    instance.graph_runner = SimpleNamespace(destroy_cuda_graphs=Mock())
    instance.tp_cpu_group = None
    instance.shutdown()

    destroy_process_group.assert_not_called()
    destroy_plugins.assert_called_once_with()
