# Windows + AMD ROCm Port — Full Requirements

Everything a user needs to reproduce this port on their own machine,
or to extend it (new GPU archs, new model adapters).

## 1. Hardware / OS

| Item | Requirement |
|---|---|
| OS | Windows 11 (tested), PowerShell 5.1+ |
| GPU | AMD RDNA3/RDNA4 consumer card (verified: RX 9070 XT, `gfx1201`, 16 GB) |
| VRAM | 8 GB min for small dense models; MoE split-load untested on Windows yet |
| RAM | 32 GB recommended |

## 2. Toolchain

| Component | Version used | Why |
|---|---|---|
| Python | 3.12 | matches prebuilt ROCm wheels |
| VS Build Tools 2022/18 | any recent | `vcvarsall.bat x64` + MSVC CRT import libs for JIT DLL linking |
| Git | any | checkout + submodules not required |

You do **not** need: CUDA Toolkit, nvcc, MSVC cl.exe working (clang replaces it),
WSL, or Linux.

## 3. Python packages (TheRock nightly stack)

**Note:** AMD support for this stack is nightly-only until ROCm 7.16 / 10.1 is formally released - pin every wheel to the same nightly stamp.

All AMD wheels must come from the SAME nightly build stamp (rocm-sdk + torch +
device modules). Fetch them via pip download from the whl-multi-arch index:

```powershell
py -3.12 -m pip download --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ `
    -d F:\ROCM-versions\cp312 "rocm[libraries,devel,device-gfx1031,device-gfx1201]"
```

Then install the local wheels (--no-deps keeps the stack consistent):

```powershell
pip install F:\ROCM-versions\cp312\*.whl --no-deps --force-reinstall

# Triton with AMD backend + tvm-ffi come from plain PyPI:
pip install triton-windows>=3.7.1 apache-tvm-ffi==0.1.13.post3
```

Install FreeToken itself without CUDA extensions:

```powershell
$env:FREETOKEN_SKIP_CUDA_EXT = "1"
pip install -e <path-to-this-repo> --no-deps --no-build-isolation
```

Or just run `dist\install.ps1`, which does all of step 3 plus step 4 automatically.

## 4. Upstream package patches (automated)

Three installed packages need small patches until merged upstream.
`dist\patch_upstream.py` applies all three **idempotently**:

| Package | Patch | Reason |
|---|---|---|
| `tvm_ffi/cpp/extension.py` | HIP-aware Windows branch: hipcc flags without `-fPIC`/MSVC-style args, `--offload-arch` emission, `amdhip64.lib` linking, host C++ built with HIP `clang++` instead of `cl.exe` | MSVC rejects FreeToken's `RuntimeCheck` pack+default-arg idiom; clang handles CUDA-C++ JIT correctly |
| `triton/backends/amd/compiler.py` | add `launch_pdl: bool = False` field to `HIPOptions` | NVIDIA-only launch kwarg otherwise crashes kernel bind on AMD |
| `uvicorn/loops/asyncio.py` | return `SelectorEventLoop` instead of `ProactorEventLoop` on win32 | `zmq.asyncio` needs `add_reader`; Proactor lacks it → silent request hang |

Optional: `TRITON_WAVE64=1` wave64 experiment patches (same file) — elementwise
kernels work, reductions are broken upstream; leave off by default.

## 5. Environment switches (runtime)

| Switch | Example value | Purpose |
|---|---|---|
| `HIP_PATH` | `G:\ROCM10RT-gfx1201` | hipcc + HIP libs for JIT builds/linking |
| `TVM_FFI_ROCM_ARCH_LIST` | `gfx1201` | tvm-ffi emits `--offload-arch=<arch>` (defaults to gfx906 → dead kernels!) |
| `TRITON_OVERRIDE_ARCH` | `gfx1201` | Triton codegen target |
| `ROCM_SDK_TARGET_FAMILY` | `gfx1201` | device family for the rocm-sdk wheel runtime (TheRock nightly only - required until ROCm 7.16 / 10.1 is formally released, after which stable wheels replace the nightlies) |
| `CC` | `<ROCM>\lib\llvm\bin\clang.EXE` | host compiler hint |
| `FREETOKEN_SKIP_CUDA_EXT` | `1` | build-time only: skip nvcc extensions |

`dist\run-server.ps1` sets all of these and calls `vcvarsall.bat` before launch
(MSVC CRT must be on `LIB` for lld-link).

## 6. Engine-side changes included in this fork

- `hip_compat.cuh`: HIP shim mapping the CUDA runtime surface (`cudaLaunchKernelEx`,
  `cudaFuncSetAttribute`, stream/error types) onto HIP; PDL attrs gated off
- PTX inline asm (`tanh.approx.f32`, `ex2.approx.f32`) gated to NVIDIA with
  `libdevice.tanh/exp2` fallbacks on AMD
- Grouped attention `BLOCK_H` padded to ≥16 (RDNA WMMA v2 requires M≥16 tiles)
- ZMQ `ipc://` → deterministic TCP loopback ports on win32 (all workers agree)
- Windows selector event-loop policy in `utils/mp.py` and the API server
- Pinned-memory fallbacks via torch when the CUDA pinned ext is absent
- `setup.py`: CUDA extensions optional (`FREETOKEN_SKIP_CUDA_EXT=1`)
- `webui/index.html`: zero-dependency browser chat client (OpenAI-compatible)

## 7. Extending this port

- **New AMD GPU**: change `gfx1201` → your arch everywhere (env vars above);
  verify WGP mode and WMMA support via `hipcc -S` probes (see DIAGNOSTICS.md)
- **New GGUF family**: add arch key to
  `python/freetoken/models/gguf/config.py::GGUF_ARCH_TO_REGISTRY`, then write an
  adapter following `models/gemma4/gguf.py` (`parse_gguf_config` + `iter_weights`
  + expert bank quant declaration). Planned: qwen35moe, laguna, deepseek4, olmoe
- Debugging methodology + failure taxonomy: [DIAGNOSTICS.md](DIAGNOSTICS.md)
