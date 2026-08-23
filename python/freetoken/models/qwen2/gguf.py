"""GGUF adapter for qwen2 models: shared dense layout, family parse_config."""
from __future__ import annotations

from freetoken.models.gguf.dense import (
    iter_dense_gguf_weights as iter_gguf_weights,
    parse_dense_gguf_config as _parse_dense,
)

from .config import parse_config


def parse_gguf_config(shim):
    return _parse_dense(shim, parse_config)


__all__ = ["parse_gguf_config", "iter_gguf_weights"]
