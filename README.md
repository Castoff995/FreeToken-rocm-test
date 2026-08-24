<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo-light.svg">
    <img alt="FreeToken" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/freetoken-logo.svg" width=65%>
  </picture>
</div>

> [!IMPORTANT]
> **This fork: native Windows + AMD ROCm port** (`FreeToken-rocm-test`).
> Verified end-to-end on an AMD Radeon RX 9070 XT (gfx1201 / RDNA4, 16 GB, Windows 11):
> `ft serve` loads dense HF safetensors models, serves OpenAI-compatible chat
> completions (including SSE token streaming) through Triton-on-AMD attention
> kernels and hipcc/tvm-ffi JIT-compiled CUDA-C++ kernels. See
> [Windows ROCm port](#windows-rocm-port) below for requirements, setup and switches.

<p align="center">
| <a href="https://www.flashml.ai/"><b>Download</b></a> | <a href="https://arxiv.org/abs/2608.16157"><b>Paper</b></a> | <a href="https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA"><b>Developer Slack</b></a> | <a href="https://discord.gg/xzwSnMdsX"><b>Community Discord</b></a> | <a href="https://github.com/FlashML-org/FreeToken/blob/main/assets/freetoken-wechatgroup.png"><b>Community WeChat</b></a> |
</p>


Unlock datacenter-class intelligence on the hardware you already own — Run 290B+ frontier MoE models locally on your gaming PC at blistering interactive speeds.

## About

FreeToken is an edge-native Mixture-of-Experts (MoE) serving engine designed for running frontier-scale open-weight models on personal and consumer hardware. It treats heterogeneous edge resources—GPUs, CPUs, host memory, and interconnects—as a unified, elastic inference platform. Its core features include:  

- **Fast Edge-Native Runtime**: Provides efficient MoE serving with bandwidth-adaptive CPU–GPU co-execution ($q^\star$ policy), full-layer double-buffered prefill streaming, global LRU expert caching, graph-compatible execution, and the FTW fast weight format.  
- **Semantic-Aware Caching**: Features semantic anchor checkpoints for recurrent state and KV caches, allowing agentic context edits (e.g., tool calls, thinking blocks) to avoid redundant context recomputation.  
- **Elastic Memory Management**: Supports dynamic, runtime VRAM re-allocation between expert caches and KV memory without engine restarts or weight reloading.  
- **Broad MoE & Ecosystem Support**: Supports frontier open-weight MoE models (e.g., DeepSeek-V4-Flash, Qwen3.6-35B-A3B, GLM-5.2) across various parameter scales and quantization formats (e.g., MXFP4, NVFP4, FP8, BF16), with Anthropic/OpenAI-compatible APIs for seamless integration with real-world coding and tool-calling agents (e.g., Codex, Claude Code, OpenCode, OpenClaw, DeepSeek Harness). 
- **Diverse Consumer Hardware**: Scales across consumer laptops, gaming desktops, and workstation GPUs, with native support for NVIDIA RTX 30, RTX 40, and RTX 50 series GPUs.  

## Getting Started

### Desktop app

Download FreeToken for Windows or Linux at [flashml.ai](https://www.flashml.ai/). It sets the engine up for you and gives you a GUI for running models, chatting, and tuning the engine.

<div align="center">
  <img alt="FreeToken Desktop" src="https://raw.githubusercontent.com/FlashML-org/FreeToken/main/assets/desktop-console.png" width=92%>
</div>

### CLI

Install FreeToken with [uv](https://docs.astral.sh/uv/) (recommended) or pip:

```bash
uv pip install "freetoken[accel]"
```

Or build from source:

```bash
git clone https://github.com/FlashML-org/FreeToken.git && cd FreeToken
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
```

For More details:

- [Install FreeToken](https://github.com/FlashML-org/FreeToken/blob/main/docs/install.md)
- [Quick start](https://github.com/FlashML-org/FreeToken/blob/main/docs/quickstart.md)
- [Supported models](https://github.com/FlashML-org/FreeToken/blob/main/docs/models.md)
- [CLI reference](https://github.com/FlashML-org/FreeToken/blob/main/docs/cli.md)

## Windows ROCm Port

This fork brings FreeToken up on **Windows 11 + AMD ROCm** with no NVIDIA toolchain.
Bring-up target: RX 9070 XT (`gfx1201`). Everything below was verified live: model load,
prefill, decode (~57 tok/s bf16 3B), SSE token streaming, and the bundled mini web UI.

## Windows ROCm / RDNA4 status

Validated custom Windows ROCm build on AMD Radeon RX 9070 XT (`gfx1201`):

- Qwen3.6 E2E serving: PASS
- ROCm 7.13
- TP=1
- Validated pin budget: 12 GiB
- Pinned: 11.854 GiB
- Fast mapped layers: 28/40
- Safe-copy layers: 12/40
- Median decode: 13.92 tok/s
- Median TTFT: 1187.6 ms
- TDR / driver reset: NO / NO

[Full Windows gfx1201 benchmark and validation report](docs/benchmarks/windows-gfx1201-qwen36.md)

This is a custom Windows ROCm/gfx1201 build validated on one RX 9070 XT system; it is not official FlashML or AMD support.

### Measured performance (RX 9070 XT, Windows, stock settings)

| Model | Precision | Fit | Decode speed | Notes |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | BF16 | full VRAM | **~72 tok/s** | stable, coherent; SSE first-token ~0.4 s |
| Qwen2.5-7B-Instruct | BF16 | `--num-pages 4096` | **~16 tok/s** | weights leave only ~1.5 GB headroom |
| gpt-oss-20b (MoE) | MXFP4 experts | `--moe-backend fused --num-pages 4096 --cuda-graph-max-bs 0` | **~12 tok/s** | stable in eager mode; see graph bug below |
| gpt-oss GGUF files | Q8_0 | - | - | rejected: GGUF loader supports `gemma4` arch only |

Decode is memory-bandwidth-bound: BF16 3B moves ~6 GB/token against ~640 GB/s,
so ~72 tok/s is near ceiling for this precision on one card. Quantized GGUF
support (planned adapters) is the main lever for large-model speed.

#### Known RDNA4 issue: MoE kernels inside CUDA-graph replay

`mxfp4_splitk_gemv` / swiglu Triton kernels run correctly eagerly but crash the
worker when executed via CUDA-graph replay on gfx1201 (dense models' graphs are
unaffected). Workaround until fixed upstream: `--cuda-graph-max-bs 0` on MoE
models. The offload backend additionally fails at capture time (`PAL failed to
finalize a command buffer`), so use `--moe-backend fused` on Windows for now.

### Quick install (automated)

A ready-made distribution kit lives in [dist/](dist/) and
[PORT_REQUIREMENTS.md](PORT_REQUIREMENTS.md):

```powershell
# once: clone + install deps, patches and freetoken
git clone https://github.com/Maxritz/FreeToken-rocm-test.git
cd FreeToken-rocm-test
powershell -ExecutionPolicy Bypass -File dist/install.ps1

# every session: engine + web UI
powershell -File dist/run-server.ps1 -Model <path-to-model>
# then open http://localhost:1420   (stop: dist/stop-server.ps1)
```

A fully portable bundle (embeddable Python + wheels, no clone needed) can be built with
`dist/make-bundle.ps1`; users then run its bundled `install.ps1` instead.

### Requirements

- Windows 11, Python 3.12, VS Build Tools (for `vcvarsall.bat` + MSVC CRT link libs)
- AMD ROCm runtime - **TheRock nightly** (`10.1.0a20260817`, HIP 7.16) until ROCm 10.1
  ships formally; set `HIP_PATH=<your-rocm-root>`
- Wheels fetched from https://rocm.nightlies.amd.com/whl-multi-arch/
- Pip stack: torch `2.15.0a0+rocm10.1.0a20260816` + `amd-torch-device-gfx1201`
  (install with `--no-deps`), `triton-windows >= 3.7.1.post27`,
  `apache-tvm-ffi == 0.1.13.post3`
- Install FreeToken itself without CUDA extensions:

```powershell
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
pip install -e <path-to-this-repo> --no-deps --no-build-isolation
```

### Environment switches

| Switch | Example value | Purpose |
|---|---|---|
| `HIP_PATH` | `<rocm-root>` | locates `hipcc`, HIP libs for JIT builds and linking |
| `TRITON_OVERRIDE_ARCH` | `gfx1201` | forces Triton codegen target |
| `TVM_FFI_ROCM_ARCH_LIST` | `gfx1201` | tvm-ffi emits `--offload-arch=<arch>` (else gfx906 default -> broken kernels) |
| `ROCM_SDK_TARGET_FAMILY` | `gfx1201` | device family for the rocm-sdk wheel runtime (nightly-only) |
| `CC` | `<rocm-root>\lib\llvm\bin\clang.EXE` | host compiler for JIT extensions |
| `FREETOKEN_SKIP_CUDA_EXT` | `1` | build-time: install without nvcc/CUDA extensions |
| `--num-pages N` | e.g. `4096` | caps KV cache pages so large dense models fit in VRAM |

Launch recipe (what `dist/run-server.ps1` does):

```bat
call "<vs>\VC\Auxiliary\Build\vcvarsall.bat" x64
set HIP_PATH=<rocm-root>
set TVM_FFI_ROCM_ARCH_LIST=gfx1201
set TRITON_OVERRIDE_ARCH=gfx1201
ft serve --model <model_path>
```

`vcvarsall` is required so the linker finds the MSVC CRT when producing JIT DLLs.

### Site-packages patches this fork relies on

Three upstream packages need small patches until merged upstream - applied automatically
by `dist/patch_upstream.py`, documented in [DIAGNOSTICS.md](DIAGNOSTICS.md):

1. **tvm_ffi/cpp/extension.py** - on Windows+HIP: use `hipcc` flags (no `-fPIC`,
   no MSVC-style `-Xcompiler` args), emit `--offload-arch`, link `amdhip64.lib`,
   and build *host* C++ with HIP `clang++` instead of `cl.exe` (MSVC rejects the
   `RuntimeCheck` pack+default-arg idiom).
2. **triton/backends/amd/compiler.py** - add `launch_pdl: bool = False` to
   `HIPOptions` so NVIDIA-only launch kwargs are accepted-and-ignored.
3. **uvicorn/loops/asyncio.py** - return `SelectorEventLoop` (not `ProactorEventLoop`)
   on win32; `zmq.asyncio` requires `add_reader`.

Engine-side patches included in this fork: HIP compat shim for CUDA-flavored kernel
headers (`hip_compat.cuh`), PTX inline-asm gated to NVIDIA with libdevice fallbacks,
WMMA-safe `BLOCK_H` padding in the grouped attention kernel, TCP loopback ZMQ
addresses with deterministic ports, Windows selector event-loop policy,
graceful CUDA-extension skipping, and a `webui/` one-file chat client.

## Status: Windows 11 + RX 9070 XT (ROCm port work log)

This fork runs natively on Windows 11 against an AMD Radeon RX 9070 XT (gfx1201,
RDNA4) using TheRock nightly ROCm runtime (`HIP_PATH`, `TVM_FFI_ROCM_ARCH_LIST`,
`TRITON_OVERRIDE_ARCH=gfx1201`, MSVC vcvars). Measured so far:
Qwen2.5-3B BF16 ~72 tok/s, Qwen2.5-7B BF16 ~15.8 tok/s, gpt-oss-20b fused-MoE
~12.2 tok/s (RDNA4 graph-replay MoE crash worked around via eager mode).

Current effort: **packed GGUF loading** for llama.cpp quant types — weights stay
quantized in VRAM (no bf16 expansion). Approach and state:

- Vendored llama.cpp quant kernels (dequant/GEMV/MMQ/MoE) already cover all
  classic, K-quant, and IQ types; the bottleneck was Python-side dispatch sets
  in `layers/gguf.py`, since widened (MMVQ = all kernel-covered types, MMQ =
  classic+K, chunked-GEMV fallback for IQ at prefill batch sizes).
- `models/gguf/dense.py` rewritten: header-only type scan, packed `.qweight`
  emission, per-layer-correct module construction (real Q4_K_M files mix
  Q4_K/Q6_K per layer), fused groups load as per-slot splits when fully
  quantized.
- Full 24-type tables in `models/gguf/dequant.py`, verified against gguf-py
  `GGML_QUANT_SIZES`; plus MXFP4 and ROCmFPX (types 100–108) dequant support.
- Adapter hooks wired into llama / qwen2 / mistral / qwen3 families; static
  validation passes against a real Mistral-7B Q4_K_M checkpoint.
- Remaining: first GPU end-to-end run of the GGUF path. The vendored kernel is
  built at runtime by PyTorch's JIT extension builder, whose Windows/HIP
  toolchain assumptions are the current source of friction (nvcc-only flags,
  hipcc wrapper arg-mangling); fixes land in `kernel/gguf.py`, with a direct
  clang or hipRTC-based loader as fallback options.
- Known RDNA4 issues parked upstream: Triton wave64 cross-lane reduction bug;
  Triton MXFP4 MoE crash under CUDA-graph replay.

## Citation

If you use FreeToken for your research, please cite our [paper](https://arxiv.org/abs/2608.16157):

```bibtex
@article{yang2026freetoken,
  title={FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution},
  author={Yang, Shuo and Fan, Xiaoze and Pan, Melissa and Xi, Haocheng and Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Han, Song and Zaharia, Matei and Xu, Chenfeng and Stoica, Ion},
  journal={arXiv preprint arXiv:2608.16157},
  year={2026}
}
```

## Acknowledgment

FreeToken was deeply inspired by [mini-sglang](https://github.com/sgl-project/mini-sglang), and
learned the design and reused code from the following projects:
[SGLang](https://github.com/sgl-project/sglang),
[vLLM](https://github.com/vllm-project/vllm),
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention),
[LightLLM](https://github.com/ModelTC/lightllm) and [llama.cpp](https://github.com/ggml-org/llama.cpp).

## License

[Apache License 2.0](https://github.com/FlashML-org/FreeToken/blob/main/LICENSE).
