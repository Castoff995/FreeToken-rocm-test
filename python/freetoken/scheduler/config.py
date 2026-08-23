from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig

def _zmq_addr(name: str) -> str:
    """patched: ipc:// is unsupported on Windows; use localhost TCP there."""
    import hashlib
    import os
    import sys

    if sys.platform == "win32":
        # patched: name-only hash - all workers must derive the SAME port
        port = 29876 + int(hashlib.sha1(name.encode()).hexdigest()[:6], 16) % 20000
        return f"tcp://127.0.0.1:{port}"
    return f"ipc:///tmp/{name}"


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return _zmq_addr("freetoken_0")

    @property
    def zmq_detokenizer_addr(self) -> str:
        return _zmq_addr("freetoken_1")

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return _zmq_addr("freetoken_2")

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True


