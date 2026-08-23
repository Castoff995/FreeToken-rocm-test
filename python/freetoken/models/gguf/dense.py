"""Shared GGUF adapter for the classic dense decoder families (llama / qwen2 /
mistral / qwen3 ...): identical gguf tensor layout (``blk.N.*``), so one config
builder + one weight iterator serve them all; each family package only wraps
its existing ``parse_config`` via a HF-config lookalike.

Weights stay in their native packed block layout: quantized tensors are yielded
as ``.qweight`` (uint8 blocks) and land in ``GGUFLinear``/``GGUFEmbedding`` modules
swapped in by :func:`convert_dense_to_gguf` (called from each family's model
constructor). Unquantized tensors (norms, biases, F32/F16/BF16 weights) flow
through the standard HF-name bf16 path - including the q/k/v -> qkv_proj and
gate/up -> gate_up_proj fusions the family loaders apply.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator

import torch

from .dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_NAME,
    dequant_any,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

    from .config import GgufConfigShim

_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}

# The projection suffixes that must share one quant type for the packed path
# (GGUF files are single-quant per tensor class; mixed projection quants are rare
# IQ mixes we don't build module layouts for).
_PROJ_SUFFIXES = (
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
)


def _proj_type_map(model_path: str) -> dict[str, int]:
    """ggml type for token_embd / output and every blk.N projection tensor.

    Cheap: GGUFReader mmaps the file; iterating only touches tensor infos, never
    the multi-GB weight data."""
    from .reader import iter_gguf_tensors

    relevant = set(_PROJ_SUFFIXES)
    types: dict[str, int] = {}
    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name in ("token_embd.weight", "output.weight"):
            types[name] = t.ggml_type
        elif name.startswith("blk.") and name.split(".", 2)[2] in relevant:
            types[name] = t.ggml_type
    return types


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
    config = parse_config(hf_like)
    _attach_gguf_types(config, shim.model_path)
    return config


def _attach_gguf_types(config: "ModelConfig", model_path: str) -> None:
    """Stash the full projection-type map on the ModelConfig (per-layer, since
    recipes like Q4_K_M vary types across layers). The attribute doubles as the
    is-gguf-packed marker for :func:`convert_dense_to_gguf`."""
    types = _proj_type_map(model_path)
    if any(t not in _UNQUANTIZED for t in types.values()):
        # ModelConfig is a frozen dataclass; gguf_types_full is an undeclared attr.
        object.__setattr__(config, "gguf_types_full", types)


def _split_modes(types: dict[str, int]) -> dict[str, bool]:
    """Per fused weight group: True -> load as per-slot splits (every layer's every
    slot quantized); False -> keep the dense bf16 fused module."""
    modes: dict[str, bool] = {}
    for fused in _WEIGHT_SLOT_MAP:
        suffixes = list(_WEIGHT_SLOT_MAP[fused].values())
        entries = [t for name, t in types.items() if name.split(".", 2)[-1] in suffixes]
        modes[fused] = bool(entries) and all(t not in _UNQUANTIZED for t in entries)
    return modes


# gguf layer-tensor suffix -> HF module-relative name. Projections keep their
# *_proj names so the loader's qkv / gate_up merge rules apply unchanged.
_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_o.weight": "self_attn.o_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.bias": "self_attn.v_proj.bias",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}


_WEIGHT_SLOT_MAP = {
    # fused param -> {slot -> expected gguf suffix}, concat order matters
    "self_attn.qkv_proj.weight": {"q": "attn_q.weight", "k": "attn_k.weight", "v": "attn_v.weight"},
    "mlp.gate_up_proj.weight": {"gate": "ffn_gate.weight", "up": "ffn_up.weight"},
}
_BIAS_SLOT_MAP = {
    "self_attn.qkv_proj.bias": {"q": "attn_q.bias", "k": "attn_k.bias", "v": "attn_v.bias"},
}


def iter_dense_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every dense-family param in GGUF naming.

    Quantized tensors keep their native packed block layout and are yielded as
    ``.qweight`` (uint8); F32/F16/BF16 tensors dequantize to bf16 under the plain
    HF name. Split-mode groups emit per-slot names (``<module>.<slot>.qweight`` /
    ``<module>.<slot>.bias``) for :class:`GgufColSplits`; plain-mode groups emit
    one fused bf16 tensor."""
    from .reader import iter_gguf_tensors

    assert include_non_moe

    types = _proj_type_map(model_path)
    split = _split_modes(types)

    # buffers for plain-mode fused groups only (bf16 fusion needs all slots)
    bufs: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

    def feed(layer: int, suffix: str, t) -> Iterator[tuple[str, torch.Tensor]]:
        quantized = t.ggml_type not in _UNQUANTIZED
        val = t.packed() if quantized else dequant_any(t)

        for fused_key, slots in _WEIGHT_SLOT_MAP.items():
            if suffix not in slots.values():
                continue
            stem = f"model.layers.{layer}." + fused_key.rsplit(".weight", 1)[0]
            slot = [k for k, v in slots.items() if v == suffix][0]
            if split[fused_key]:
                yield f"{stem}.{slot}.qweight", val
                return
            buf = bufs.setdefault(layer, {}).setdefault(fused_key, {})
            buf[slot] = val
            if len(buf) == len(slots):
                del bufs[layer][fused_key]
                if not bufs[layer]:
                    del bufs[layer]
                yield stem + ".weight", torch.cat([buf[k] for k in slots], dim=0).contiguous()
            return

        for fused_key, slots in _BIAS_SLOT_MAP.items():
            if suffix not in slots.values():
                continue
            stem = f"model.layers.{layer}." + fused_key.rsplit(".bias", 1)[0]
            slot = [k for k, v in slots.items() if v == suffix][0]
            if split[fused_key.replace(".bias", ".weight")]:
                yield f"{stem}.{slot}.bias", val
                return
            buf = bufs.setdefault(layer, {}).setdefault(fused_key, {})
            buf[slot] = val
            if len(buf) == len(slots):
                del bufs[layer][fused_key]
                if not bufs[layer]:
                    del bufs[layer]
                yield stem + ".bias", torch.cat([buf[k] for k in slots], dim=0).contiguous()
            return

        rel = _SUFFIX_MAP.get(suffix)
        if rel is not None:
            if quantized:
                yield f"model.layers.{layer}." + rel.replace(".weight", ".qweight"), val
            else:
                yield f"model.layers.{layer}.{rel}", val

    try:
        for t in iter_gguf_tensors(model_path):
            name = t.name
            quantized = t.ggml_type not in _UNQUANTIZED
            if name == "token_embd.weight":
                yield (
                    "model.embed_tokens.qweight"
                    if quantized
                    else "model.embed_tokens.weight",
                    t.packed() if quantized else dequant_any(t),
                )
            elif name == "output_norm.weight":
                yield "model.norm.weight", dequant_any(t)
            elif name == "output.weight":
                yield (
                    "lm_head.qweight" if quantized else "lm_head.weight",
                    t.packed() if quantized else dequant_any(t),
                )
            elif name.startswith("blk."):
                if "exps" in name:
                    raise NotImplementedError(
                        f"{name}: routed-expert tensors need a MoE-specific GGUF adapter"
                    )
                suffix = name.split(".", 2)[2]
                layer = int(name.split(".")[1])
                yield from feed(layer, suffix, t)
    finally:
        leftovers = {k: sorted(v) for k, v in bufs.items()} if bufs else None
        assert not leftovers, f"incomplete fused groups: {leftovers}"


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def convert_dense_to_gguf(model, config: "ModelConfig") -> None:
    """In place: replace dense projections (+ embedding / lm head when the checkpoint
    quantizes them) with native GGUF ops holding packed block weights.

    Fused projections load as per-slot :class:`GgufColSplits` whenever every slot is
    quantized (types may vary per layer, e.g. Q4_K_M); otherwise the dense bf16
    module stays and the loader supplies fused bf16 weights. No-op unless
    :func:`_attach_gguf_types` marked the config."""
    from freetoken.distributed import get_tp_info

    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear, GgufColSplits

    types = getattr(config, "gguf_types_full", None)
    if not types:
        return
    assert get_tp_info().size == 1, (
        "packed-GGUF dense models support TP=1 only (quant blocks are not sharded)"
    )

    split = _split_modes(types)

    def layer_type(layer_id: int, suffix: str) -> int | None:
        return types.get(f"blk.{layer_id}.{suffix}")

    def swap(owner, attr, layer_id, group_key):
        """Fused module -> per-slot GgufColSplits (split mode only; plain-mode
        groups keep their dense module and get fused bf16 weights)."""
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        slots = _WEIGHT_SLOT_MAP[group_key]
        parts = []
        for slot, dim in zip(slots, _split_dims(attr, config)):
            qt = layer_type(layer_id, slots[slot])
            assert qt is not None and qt not in _UNQUANTIZED, (
                f"blk.{layer_id}.{slots[slot]} missing/unquantized in split mode"
            )
            parts.append((slot, dim, qt))
        setattr(
            owner,
            attr,
            GgufColSplits(in_features, parts, has_bias=lin.bias is not None),
        )

    def swap_single(owner, attr, layer_id, suffix):
        lin = getattr(owner, attr)
        out_features, in_features = lin.weight.shape
        qt = layer_type(layer_id, suffix)
        if qt is None or qt in _UNQUANTIZED:
            return  # unquantized -> keep dense module, loader yields bf16 .weight
        setattr(
            owner,
            attr,
            GGUFLinear(in_features, out_features, qt, has_bias=lin.bias is not None),
        )

    inner = model.model
    qkv_key = "self_attn.qkv_proj.weight"
    gate_up_key = "mlp.gate_up_proj.weight"
    for lid, layer in enumerate(inner.layers.op_list):
        if split[qkv_key]:
            swap(layer.self_attn, "qkv_proj", lid, qkv_key)
        if split[gate_up_key]:
            swap(layer.mlp, "gate_up_proj", lid, gate_up_key)
        swap_single(layer.self_attn, "o_proj", lid, "attn_output.weight")
        swap_single(layer.mlp, "down_proj", lid, "ffn_down.weight")

    embed_type = types.get("token_embd.weight")
    if embed_type is not None and embed_type not in _UNQUANTIZED:
        inner.embed_tokens = GGUFEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_type=embed_type,
        )
        if config.tie_word_embeddings:
            from freetoken.models.gemma4.gguf import GGUFTiedLMHead

            model.lm_head = GGUFTiedLMHead(inner.embed_tokens, embed_type)

    output_type = types.get("output.weight")
    if output_type is not None and output_type not in _UNQUANTIZED and not config.tie_word_embeddings:
        model.lm_head = GGUFLinear(config.hidden_size, config.vocab_size, output_type)


def _split_dims(attr: str, config: "ModelConfig") -> list[int]:
    """Per-slot output widths of a fused projection at TP=1."""
    if attr == "qkv_proj":
        return [
            config.num_qo_heads * config.head_dim,
            config.num_kv_heads * config.head_dim,
            config.num_kv_heads * config.head_dim,
        ]
    assert attr == "gate_up_proj"
    return [config.intermediate_size, config.intermediate_size]


def maybe_convert_dense_to_gguf(model, config: "ModelConfig") -> None:
    """Hook for family model constructors: swap when loading a packed GGUF."""
    if getattr(config, "gguf_types_full", None):
        convert_dense_to_gguf(model, config)


__all__ = [
    "parse_dense_gguf_config",
    "iter_dense_gguf_weights",
    "convert_dense_to_gguf",
    "maybe_convert_dense_to_gguf",
]
