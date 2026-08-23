"""Shared GGUF adapter for the classic dense decoder families (llama / qwen2 /
mistral / qwen3 ...): identical gguf tensor layout (``blk.N.*``), so one config
builder + one weight iterator serve them all; each family package only wraps
its existing ``parse_config`` via a HF-config lookalike.

Weights are fully dequantized to bf16 at load (any ggml quant via
``dequant_any``), then flow through the standard HF-name path - including the
q/k/v -> qkv_proj and gate/up -> gate_up_proj fusions the family loaders apply.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from .dequant import dequant_any

if TYPE_CHECKING:
    from .config import GgufConfigShim


def _rope_scaling(m: dict, prefix: str) -> dict | None:
    stype = m.get(f"{prefix}.rope.scaling.type")
    if stype in (None, "none"):
        return None
    factor = m.get(f"{prefix}.rope.scaling.factor", 1.0)
    scaling: dict = {"rope_type": str(stype), "factor": float(factor)}
    if stype == "yarn":
        orig = m.get(f"{prefix}.rope.scaling.original_max_position_embeddings")
        if orig is not None:
            scaling["original_max_position_embeddings"] = int(orig)
    if stype == "llama3":
        lo = m.get(f"{prefix}.rope.scaling.low_freq_factor")
        hi = m.get(f"{prefix}.rope.scaling.high_freq_factor")
        if lo is not None:
            scaling["low_freq_factor"] = float(lo)
        if hi is not None:
            scaling["high_freq_factor"] = float(hi)
    return scaling


def parse_dense_gguf_config(shim: "GgufConfigShim", parse_config):
    """Build the family ``ModelConfig`` by feeding a HF-config lookalike (built
    from GGUF KV metadata) to the family's own ``parse_config``."""
    m = shim.metadata
    prefix = shim.model_type  # metadata keys are namespaced by arch, e.g. "llama."

    def g(key: str, default=None):
        return m.get(f"{prefix}.{key}", default)

    head_count = int(g("attention.head_count"))
    kv_heads = g("attention.head_count_kv")
    num_kv_heads = int(kv_heads) if not isinstance(kv_heads, list) else int(kv_heads[0])
    key_length = g("attention.key_length")

    hf_like = SimpleNamespace(
        num_hidden_layers=int(g("block_count")),
        num_attention_heads=head_count,
        num_key_value_heads=num_kv_heads,
        head_dim=int(key_length) if key_length else None,
        hidden_size=int(g("embedding_length")),
        intermediate_size=int(g("feed_forward_length")),
        rms_norm_eps=float(g("attention.layer_norm_rms_epsilon", 1e-5)),
        max_position_embeddings=int(g("context_length", 4096)),
        rope_theta=float(g("rope.freq_base", 10_000.0)),
        rope_scaling=_rope_scaling(m, prefix),
        hidden_act="silu",
        vocab_size=int(shim.vocab_size),
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        torch_dtype="bfloat16",
    )
    return parse_config(hf_like)


# gguf layer-tensor suffix -> HF module-relative name. Projections keep their
# *_proj names so the loader's qkv / gate_up merge rules apply unchanged.
_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_o.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


_FUSE_GROUPS = {
    # fused param suffix -> {slot -> expected gguf suffix}, concat order matters
    "self_attn.qkv_proj.weight": {"q": "attn_q.weight", "k": "attn_k.weight", "v": "attn_v.weight"},
    "self_attn.qkv_proj.bias": {"q": "attn_q.bias", "k": "attn_k.bias", "v": "attn_v.bias"},
    "mlp.gate_up_proj.weight": {"gate": "ffn_gate.weight", "up": "ffn_up.weight"},
}


def iter_dense_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, bf16 tensor) for every dense-family param in GGUF naming."""
    from .reader import iter_gguf_tensors

    assert include_non_moe

    # fusion buffers: layer -> {fused_suffix -> {slot: bf16 tensor}}
    bufs: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    def feed(layer: int, suffix: str, t) -> Iterator[tuple[str, torch.Tensor]]:
        for fused, slots in _FUSE_GROUPS.items():
            if suffix in slots.values():
                buf = bufs.setdefault(layer, {}).setdefault(fused, {})
                buf[[k for k, v in slots.items() if v == suffix][0]] = dequant_any(t)
                if len(buf) == len(slots):
                    del bufs[layer][fused]
                    if not bufs[layer]:
                        del bufs[layer]
                    order = list(slots.keys())
                    yield (
                        f"model.layers.{layer}.{fused}",
                        torch.cat([buf[k] for k in order], dim=0).contiguous(),
                    )
                return
        rel = _SUFFIX_MAP.get(suffix)
        if rel is not None:
            yield f"model.layers.{layer}.{rel}", dequant_any(t)

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.weight", dequant_any(t)
        elif name == "output_norm.weight":
            yield "model.norm.weight", dequant_any(t)
        elif name == "output.weight":
            yield "lm_head.weight", dequant_any(t)
        elif name.startswith("blk."):
            if "exps" in name:
                raise NotImplementedError(
                    f"{name}: routed-expert tensors need a MoE-specific GGUF adapter"
                )
            suffix = name.split(".", 2)[2]
            layer = int(name.split(".")[1])
            yield from feed(layer, suffix, t)


__all__ = ["parse_dense_gguf_config", "iter_dense_gguf_weights"]
