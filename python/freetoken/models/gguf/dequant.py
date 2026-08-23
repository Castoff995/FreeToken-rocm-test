"""GGML block-quant dequantization in pure torch (the formats this repo's GGUF
checkpoints use: Q4_0, Q6_K, plus trivial F32/F16/BF16).

This is the *reference / CPU* path, NOT the engine's hot path: GGUF weights stay
packed and are dequantized inside the borrowed ggml CUDA kernels (see
``freetoken.kernel.gguf``). These routines are used only to (a) materialize the few
dense F32/F16 tensors at load (norms, scales, router) via :func:`dequantize`, and
(b) cross-check the CUDA kernels in tests. The ``BLOCK_SHAPE`` table and
:func:`row_bytes` are the type metadata the packed (kernel) path also relies on.

Each ``dequant_*`` takes the raw little-endian bytes as a ``uint8`` tensor whose
final axis spans whole blocks, and returns the values in *storage order* (ggml's
fastest axis first); the caller reshapes to the torch shape (``dims[::-1]``). The
math mirrors ``ggml-quants.c``.
"""

from __future__ import annotations

import torch

# ggml_type enum values (subset present in these checkpoints).
GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8
GGML_Q6_K = 14
GGML_BF16 = 30
GGML_MXFP4 = 39
GGML_Q4_0_ROCMFP4 = 100
GGML_Q4_0_ROCMFP4_FAST = 101
GGML_Q4_0_ROCMI4 = 108

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q8_0: (32, 34),
    GGML_Q6_K: (256, 210),
    # OCP MXFP4: E8M0 scale byte + 16 packed nibbles (llama.cpp block_mxfp4)
    GGML_MXFP4: (32, 17),
    # ROCmFPX family (charlie12345/ROCmFPX): 16 packed nibbles + UE4M3 scale(s)
    GGML_Q4_0_ROCMFP4: (32, 18),
    GGML_Q4_0_ROCMFP4_FAST: (32, 17),
    GGML_Q4_0_ROCMI4: (32, 17),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q8_0: "Q8_0",
    GGML_Q6_K: "Q6_K",
    GGML_MXFP4: "MXFP4",
    GGML_Q4_0_ROCMFP4: "Q4_0_ROCMFP4",
    GGML_Q4_0_ROCMFP4_FAST: "Q4_0_ROCMFP4_FAST",
    GGML_Q4_0_ROCMI4: "Q4_0_ROCMI4",
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


def _f16_scales(raw: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Reinterpret bytes ``[lo:hi]`` (2 per block) of each block row as fp16 -> fp32 [N,1]."""
    return raw[:, lo:hi].contiguous().view(torch.float16).to(torch.float32)


def dequant_q4_0(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0: per 32-elem block = fp16 scale ``d`` + 16 packed nibbles; ``w = d*(q-8)``.

    Byte ``j`` of the 16 holds element ``j`` in its low nibble and ``j+16`` in its high
    nibble, so storage order within the block is ``[lo0..lo15, hi0..hi15]``.
    """
    raw = raw.reshape(-1, 18)
    d = _f16_scales(raw, 0, 2)  # [N,1]
    qs = raw[:, 2:18]  # [N,16] uint8
    lo = (qs & 0x0F).to(torch.float32)
    hi = (qs >> 4).to(torch.float32)
    q = torch.cat([lo, hi], dim=1)  # [N,32]
    return ((q - 8.0) * d).reshape(-1).to(out_dtype)


def dequant_q6_k(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q6_K: 256-elem super-block = 128B low nibbles + 64B high 2-bits + 16 int8
    sub-scales + fp16 ``d``. Direct vectorization of ggml's two-half loop."""
    raw = raw.reshape(-1, 210)
    n = raw.shape[0]
    ql = raw[:, 0:128]  # [n,128]
    qh = raw[:, 128:192]  # [n,64]
    sc = raw[:, 192:208].view(torch.int8).to(torch.float32)  # [n,16]
    d = _f16_scales(raw, 208, 210)  # [n,1]

    y = torch.empty((n, 256), dtype=torch.float32, device=raw.device)
    # l in 0..15 -> is=0; l in 16..31 -> is=1 (per ggml: is = l/16).
    is_idx = (torch.arange(32, device=raw.device) // 16)  # [32] in {0,1}
    for h in range(2):  # two 128-elem halves of the super-block
        qlh = ql[:, h * 64:(h + 1) * 64]  # [n,64]
        qhh = qh[:, h * 32:(h + 1) * 32]  # [n,32]
        sch = sc[:, h * 8:(h + 1) * 8]  # [n,8]
        a = qlh[:, 0:32].to(torch.int32)  # ql[l]
        b = qlh[:, 32:64].to(torch.int32)  # ql[l+32]
        hb = qhh.to(torch.int32)  # qh[l]
        q1 = ((a & 0x0F) | (((hb >> 0) & 3) << 4)) - 32
        q2 = ((b & 0x0F) | (((hb >> 2) & 3) << 4)) - 32
        q3 = ((a >> 4) | (((hb >> 4) & 3) << 4)) - 32
        q4 = ((b >> 4) | (((hb >> 6) & 3) << 4)) - 32
        s1 = sch.index_select(1, is_idx + 0).to(torch.float32)
        s2 = sch.index_select(1, is_idx + 2).to(torch.float32)
        s3 = sch.index_select(1, is_idx + 4).to(torch.float32)
        s4 = sch.index_select(1, is_idx + 6).to(torch.float32)
        base = h * 128
        y[:, base + 0:base + 32] = d * s1 * q1.to(torch.float32)
        y[:, base + 32:base + 64] = d * s2 * q2.to(torch.float32)
        y[:, base + 64:base + 96] = d * s3 * q3.to(torch.float32)
        y[:, base + 96:base + 128] = d * s4 * q4.to(torch.float32)
    return y.reshape(-1).to(out_dtype)


# --------------------------------------------------------------------------------------
# MXFP4 (ggml type 39) and the ROCmFPX family (types 100/101/108, from
# charlie12345/ROCmFPX). All are 32-element blocks of packed nibbles with an
# 8-bit exponent scale; they differ only in codebook + scale encoding.
# --------------------------------------------------------------------------------------

# E2M1-derived codebooks: index = low 3 bits, sign = bit 3.
_KV_MXFP4 = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
_KV_ROCMFP4 = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]


_MXFP4_LUT = torch.tensor(
    [_KV_MXFP4[i] if i < 8 else -_KV_MXFP4[i - 8] for i in range(16)], dtype=torch.float32
)
_ROCMFP4_LUT = torch.tensor(
    [_KV_ROCMFP4[i] if i < 8 else -_KV_ROCMFP4[i - 8] for i in range(16)], dtype=torch.float32,
)


def _ue4m3_half_table() -> torch.Tensor:
    """127-entry UE4M3-half scale table from rocmfp4.c: byte b = e_field*8+m;
    e_field==0 -> subnormal m*2^-10; else (8+m)*2^(e_field-11). Byte >= 0x7f -> 0."""
    t = torch.empty(128, dtype=torch.float32)
    for b in range(127):
        e_field, m = divmod(b, 8)
        t[b] = m * 2.0**-10 if e_field == 0 else (8 + m) * 2.0 ** (e_field - 11)
    t[127] = 0.0
    return t


_UE4M3_HALF = _ue4m3_half_table()


def _nibble_blocks(raw: torch.Tensor, scale_cols: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Split [N, 17|18] block rows into codes [N,32] and per-half scales [N,2]."""
    qs = raw[:, :16]
    scales = raw[:, scale_cols].to(torch.int64)
    lo = (qs & 0x0F).to(torch.int64)
    hi = (qs >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=2)  # [N, 16, 2]
    return codes, scales


def dequant_mxfp4(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """block_mxfp4 { uint8 e(E8M0); uint8 qs[16]; } -> d = 2^(e-127)."""
    n = raw.shape[0]
    e = raw[:, 0].to(torch.int64)
    qs = raw[:, 1:]
    lo = (qs & 0x0F).to(torch.int64)
    hi = (qs >> 4).to(torch.int64)
    d = torch.pow(2.0, (e - 127).to(torch.float32))  # E8M0 half-bias
    out = raw.new_empty((n, 32), dtype=torch.float32)
    out[:, :16] = _MXFP4_LUT[lo] * d[:, None]
    out[:, 16:] = _MXFP4_LUT[hi] * d[:, None]
    return out.to(out_dtype).reshape(-1)


def dequant_q4_0_rocmfp4(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0_ROCMFP4 { qs[16]; e[2](UE4M3-half per 16-weight half block); }"""
    n = raw.shape[0]
    codes, scales = _nibble_blocks(raw, [16, 17])
    s = _UE4M3_HALF[scales]  # [N, 2]
    out = raw.new_empty((n, 32), dtype=torch.float32)
    out[:, :16] = _ROCMFP4_LUT[codes[:, :, 0]] * s[:, 0:1]
    out[:, 16:] = _ROCMFP4_LUT[codes[:, :, 1]] * s[:, 1:2]
    return out.to(out_dtype).reshape(-1)


def dequant_q4_0_rocmfp4_fast(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0_ROCMFP4_FAST { qs[16]; e(UE4M3-half for the whole 32-block); }"""
    n = raw.shape[0]
    codes, scales = _nibble_blocks(raw, [16])
    s = _UE4M3_HALF[scales[:, 0]]  # [N]
    out = raw.new_empty((n, 32), dtype=torch.float32)
    out[:, :16] = _ROCMFP4_LUT[codes[:, :, 0]] * s[:, None]
    out[:, 16:] = _ROCMFP4_LUT[codes[:, :, 1]] * s[:, None]
    return out.to(out_dtype).reshape(-1)


def dequant_q4_0_rocmi4(raw: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Q4_0_ROCMI4 { int8 qs as two's-complement nibbles; e(UE4M3-half); }"""
    n = raw.shape[0]
    qs = raw[:, :16].to(torch.int64)
    lo = (qs & 0x0F)
    hi = (qs >> 4)
    lo = torch.where(lo >= 8, lo - 16, lo).to(torch.float32)
    hi = torch.where(hi >= 8, hi - 16, hi).to(torch.float32)
    s = _UE4M3_HALF[raw[:, 16].to(torch.int64)]  # [N]
    out = raw.new_empty((n, 32), dtype=torch.float32)
    out[:, :16] = lo * s[:, None]
    out[:, 16:] = hi * s[:, None]
    return out.to(out_dtype).reshape(-1)



_DEQUANT = {
    GGML_Q4_0: dequant_q4_0,
    GGML_Q6_K: dequant_q6_k,
    GGML_MXFP4: dequant_mxfp4,
    GGML_Q4_0_ROCMFP4: dequant_q4_0_rocmfp4,
    GGML_Q4_0_ROCMFP4_FAST: dequant_q4_0_rocmfp4_fast,
    GGML_Q4_0_ROCMI4: dequant_q4_0_rocmi4,
}


def dequantize(raw: torch.Tensor, ggml_type: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize ``raw`` (uint8) of any supported ggml type to flat ``out_dtype``."""
    if ggml_type == GGML_F32:
        return raw.view(torch.float32).to(out_dtype)
    if ggml_type == GGML_F16:
        return raw.view(torch.float16).to(out_dtype)
    if ggml_type == GGML_BF16:
        return raw.view(torch.bfloat16).to(out_dtype)
    fn = _DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(
            f"dequant for ggml type {GGML_NAME.get(ggml_type, ggml_type)} not implemented"
        )
    return fn(raw, out_dtype)


__all__ = [
    "GGML_F32",
    "GGML_F16",
    "GGML_BF16",
    "GGML_Q4_0",
    "GGML_Q8_0",
    "GGML_Q6_K",
    "GGML_NAME",
    "BLOCK_SHAPE",
    "row_bytes",
    "dequant_q4_0",
    "dequant_q6_k",
    "dequantize",
]


def dequant_any(t, out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize any GgufTensor to a flat bf16 tensor in torch storage order.

    Tries this repo's pure-torch table first (Q4_0 / Q8_0 / Q6_K / F16 / BF16 /
    F32); anything else (Q4_K, Q5_K, Q2_K/Q3_K, IQ quants ...) falls back to the
    ``gguf`` package's numpy reference dequantizer. Load-time only - the result
    is materialized bf16, so the engine's hot path never sees these formats.
    """
    gt = int(t.ggml_type)
    try:
        return dequantize(t.packed().reshape(-1), gt, out_dtype).reshape(t.shape)
    except (KeyError, NotImplementedError):
        import gguf
        import numpy as np

        qt = gguf.GGMLQuantizationType(gt)
        vals = gguf.dequantize(t.packed().numpy(), qt)
        numel = 1
        for d in t.shape:
            numel *= d
        assert vals.size == numel, f"{vals.size} != {numel} for {t.name}"
        return torch.from_numpy(vals.reshape(t.shape)).to(out_dtype)


