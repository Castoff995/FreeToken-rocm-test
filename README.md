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
prefill, decode (~57 tok/s bf16 3B), SSE streaming, and the bundled mini web UI.

### What works today

| Area | Status |
|---|---|
| Dense HF safetensors models (bf16/fp16), single GPU | ✅ working (Qwen2.5-3B verified; 7B fits only with `--num-pages` cap) |
| OpenAI-compatible `/v1/chat/completions`, streaming | ✅ working |
| Triton attention/norm/activation kernels on RDNA4 | ✅ working (AMD backend) |
| tvm-ffi JIT CUDA kernels (`index`, `store`, `radix`) | ✅ compiling via `hipcc` + host `clang++` |
| Web chat UI | ✅ `G:\FreeToken\webui\index.html` (serve on port 1420) |
| CUDA extensions skipped gracefully | ✅ `FREETOKEN_SKIP_CUDA_EXT=1` |
| Pinned-memory / ZMQ IPC fallbacks on Windows | ✅ TCP loopback + torch pin_memory |
| MoE expert offload (split CPU–GPU execution) | ⚠️ untested on Windows yet |
| GGUF loader | ⚠️ `gemma4` architecture only — Qwen/Laguna/DeepSeek adapters pending |
| ROCmFPX-quantized GGUFs | ❌ proprietary quant types, needs the source fork |

### Quick install (automated)

A ready-made distribution kit lives in [dist/](dist/) and
[PORT_REQUIREMENTS.md](PORT_REQUIREMENTS.md):

\\powershell
powershell -ExecutionPolicy Bypass -File dist\install.ps1        # deps + patches + freetoken
powershell -File dist\run-server.ps1 -Model G:\models\my-model   # engine + web UI
# then open http://localhost:1420  (stop: dist\stop-server.ps1)
\
### Requirements

- Windows 11, Python 3.12, VS Build Tools (for `vcvarsall.bat` + MSVC CRT link libs)
- AMD ROCm runtime (TheRock nightly used here): `HIP_PATH=G:\ROCM10RT-gfx1201`
- Pip stack (matching TheRock nightly `10.1.0a20260817`):
  - `torch 2.15.0a0+rocm10.1.0a20260816` + `amd_torch_device_gfx1201` (install with `--no-deps`)
  - `triton-windows >= 3.7.1.post27` (ships the `amd` backend)
  - `apache-tvm-ffi == 0.1.13.post3`
- Install FreeToken itself without CUDA extensions:

```powershell
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
pip install -e G:\FreeToken --no-deps --no-build-isolation
```

### Environment switches

| Switch | Value | Purpose |
|---|---|---|
| `HIP_PATH` | `G:\ROCM10RT-gfx1201` | locates `hipcc`, HIP libs for JIT builds & linking |
| `TRITON_OVERRIDE_ARCH` | `gfx1201` | forces Triton codegen target |
| `TVM_FFI_ROCM_ARCH_LIST` | `gfx1201` | tvm-ffi emits `--offload-arch=gfx1201` (else gfx906 default → broken kernels) |
| `CC` | `<ROCM>\lib\llvm\bin\clang.EXE` | host compiler for JIT extensions |
| `FREETOKEN_SKIP_CUDA_EXT` | `1` | build-time: install without nvcc/CUDA extensions |
| `--num-pages N` | e.g. `4096` | caps KV cache pages so large dense models fit in VRAM |

Launch recipe (see `run_serve.cmd` pattern):

```bat
call "...\VC\Auxiliary\Build\vcvarsall.bat" x64
set HIP_PATH=G:\ROCM10RT-gfx1201
set TVM_FFI_ROCM_ARCH_LIST=gfx1201
set TRITON_OVERRIDE_ARCH=gfx1201
ft serve --model <model_path>
```

`vcvarsall` is required so the linker finds the MSVC CRT when producing JIT DLLs.

### Site-packages patches this fork relies on

Three upstream packages need small patches until merged upstream (all documented in
[DIAGNOSTICS.md](DIAGNOSTICS.md)):

1. **tvm_ffi/cpp/extension.py** — on Windows+HIP: use `hipcc` flags (no `-fPIC`,
   no MSVC-style `-Xcompiler` args), emit `--offload-arch`, link `amdhip64.lib`,
   and build *host* C++ with HIP `clang++` instead of `cl.exe` (MSVC rejects the
   `RuntimeCheck` pack+default-arg idiom).
2. **triton/backends/amd/compiler.py** — add `launch_pdl: bool = False` to
   `HIPOptions` so NVIDIA-only launch kwargs are accepted-and-ignored.
3. **uvicorn/loops/asyncio.py** — return `SelectorEventLoop` (not `ProactorEventLoop`)
   on win32; `zmq.asyncio` requires `add_reader`.

Engine-side patches included in this fork: HIP compat shim for CUDA-flavored kernel
headers (`hip_compat.cuh`), PTX inline-asm gated to NVIDIA with libdevice fallbacks,
WMMA-safe `BLOCK_H` padding in the grouped attention kernel, TCP loopback ZMQ
addresses with deterministic ports, Windows selector event-loop policy,
graceful CUDA-extension skipping, and a `webui/` one-file chat client.

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
