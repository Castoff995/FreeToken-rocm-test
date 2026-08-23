"""Native-GGUF quantized layers: weights stay in their packed block layout and are
dequantized *inside* the borrowed llama.cpp kernels (no bf16 copy ever materialized).

Mirrors vLLM/sglang's ``GGUFLinearMethod`` / ``GGUFEmbeddingMethod`` dispatch, ported
onto FreeToken's ``BaseOP``. FreeToken keeps fused projections (qkv, gate_up) as a
single tensor: because Q4_0/K-quants pack each *output row* independently over the
input dim, the loader can concatenate the per-shard packed rows along dim 0 (they
share an input dim, hence the same ``row_bytes``), so a fused layer is still one
``[out, row_bytes]`` qweight -- no per-shard padding bookkeeping needed.

TP is assumed to be 1 (the gemma4 GGUF path restricts to TP=1, like the HF path).
"""

from __future__ import annotations

import torch

from freetoken.models.gguf.dequant import (
    BLOCK_SHAPE,
    GGML_BF16,
    GGML_F16,
    GGML_F32,
    GGML_IQ1_M,
    GGML_IQ1_S,
    GGML_IQ2_S,
    GGML_IQ2_XS,
    GGML_IQ2_XXS,
    GGML_IQ3_S,
    GGML_IQ3_XXS,
    GGML_IQ4_NL,
    GGML_IQ4_XS,
    GGML_NAME,
    GGML_Q2_K,
    GGML_Q3_K,
    GGML_Q4_0,
    GGML_Q4_1,
    GGML_Q4_K,
    GGML_Q5_0,
    GGML_Q5_1,
    GGML_Q5_K,
    GGML_Q6_K,
    GGML_Q8_0,
    row_bytes,
)

from .base import BaseOP

# ggml type groups for kernel dispatch — everything the vendored ggml kernels
# implement (see csrc/gguf/gguf_kernel.cu case labels). Weights stay packed for
# all of these; nothing here dequantizes to bf16 at load time.
_UNQUANTIZED = {GGML_F32, GGML_F16, GGML_BF16}
# MMVQ GEMV kernels exist for every quant type the kernels cover.
_MMVQ = {
    GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
    GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
    GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ3_XXS, GGML_IQ1_S, GGML_IQ4_NL,
    GGML_IQ3_S, GGML_IQ2_S, GGML_IQ4_XS, GGML_IQ1_M,
}
# MMQ (large-batch tiled) kernels: classic + K-quants only.
_MMQ = {GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0,
        GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K}
# CPU/GPU dequantize fallback (per-call, never at load).
_DEQUANT = _MMVQ

# Below this token count, the MMVQ GEMV kernel wins (matches vLLM's heuristic).
_MMVQ_SAFE = 6


def fused_mul_mat_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    """y = x @ dequant(qweight).T, dispatched by batch size and quant type."""
    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    out_features = qweight.shape[0]
    if x.shape[0] == 0:
        return x.new_empty((0, out_features))
    if qweight_type in _UNQUANTIZED:
        return x @ qweight.T
    if x.shape[0] <= _MMVQ_SAFE and qweight_type in _MMVQ:
        return ggml_mul_mat_vec_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMQ:
        return ggml_mul_mat_a8(qweight, x, qweight_type, out_features)
    if qweight_type in _MMVQ:
        # no MMQ kernel for this type: chunk the batch through the GEMV kernel
        chunks = [
            ggml_mul_mat_vec_a8(qweight, x[i : i + _MMVQ_SAFE], qweight_type, out_features)
            for i in range(0, x.shape[0], _MMVQ_SAFE)
        ]
        return torch.cat(chunks) if len(chunks) > 1 else chunks[0]
    if qweight_type in _DEQUANT:
        block, type_size = BLOCK_SHAPE[qweight_type]
        in_features = qweight.shape[1] // type_size * block
        weight = ggml_dequantize(qweight, qweight_type, out_features, in_features, x.dtype)
        return x @ weight.T
    raise NotImplementedError(f"unsupported GGUF type {GGML_NAME.get(qweight_type, qweight_type)}")


class GGUFLinear(BaseOP):
    """Linear whose weight is a native GGUF block-quantized ``[out, row_bytes]`` tensor."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_type: int,
        has_bias: bool = False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self._quant_type = quant_type
        self.qweight = torch.empty(out_features, row_bytes(in_features, quant_type), dtype=torch.uint8)
        self.bias = torch.empty(out_features) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = fused_mul_mat_gguf(x, self.qweight, self._quant_type)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class GgufColSplits(BaseOP):
    """Column-split GGUF linears for fused modules whose slots carry *different*
    quant types (Q4_K_M's Q6_K attn_v/ffn_down over a Q4_K body). Holds one
    ``GGUFLinear`` per slot under the slot name; forward concatenates the slot
    outputs so it drops in for ``LinearQKVMerged``/``LinearColParallelMerged``."""

    def __init__(
        self,
        in_features: int,
        parts: list[tuple[str, int, int]],
        has_bias: bool = False,
    ):
        # parts: [(slot_name, out_features, ggml_type)] in concat order.
        self._order = [name for name, _, _ in parts]
        for name, out_features, quant_type in parts:
            setattr(
                self,
                name,
                GGUFLinear(in_features, out_features, quant_type, has_bias=has_bias),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([getattr(self, n).forward(x) for n in self._order], dim=-1)


class GGUFEmbedding(BaseOP):
    """Vocab embedding stored as a native GGUF block-quantized table.

    The full table is never dequantized: only the looked-up rows are gathered (in
    packed form) and dequantized per lookup, matching vLLM's ``_apply_gguf_embedding``.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        quant_type: int,
        embed_scale: float | None = None,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.gguf import ggml_dequantize

        flat = x.flatten()
        rows = self.qweight.index_select(0, flat)  # [n, row_bytes] packed
        y = ggml_dequantize(rows, self._quant_type, flat.shape[0], self.embedding_dim, torch.bfloat16)
        y = y.view(*x.shape, self.embedding_dim)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(self._embed_scale, dtype=y.dtype, device=y.device)
            y = y * self._embed_scale_t
        return y


__all__ = ["GGUFLinear", "GGUFEmbedding", "fused_mul_mat_gguf"]
