# Custom Windows ROCm gfx1201 build

This document describes the repository-controlled custom build validated on an AMD Radeon
RX 9070 XT (`gfx1201`). It is not a statement of official upstream FreeToken support.

## Tested stack

- Windows 11 and Python 3.12
- `torch==2.11.0+rocm7.13.0`, HIP `7.13.99004`
- official AMD ROCm SDK 7.13, `gfx120X-all`
- `triton-windows==3.7.1.post27` (imports as Triton `3.7.1`)
- `apache-tvm-ffi==0.1.13.post3`
- TP=1

Do not use the experimental ROCm 10.1 nightly environment: it produced
`hipErrorInvalidKernelFile` on the first executable GPU kernel on the tested machine.

## Fresh installation

Run these commands from a PowerShell prompt. Replace the repository and environment paths as
needed; the setup script itself derives SDK paths from the active environment and does not
contain a fixed user home path.

```powershell
py -3.12 -m venv C:\Users\<user>\FreeToken-rocm-finaltest\.venv
$python = "C:\Users\<user>\FreeToken-rocm-finaltest\.venv\Scripts\python.exe"

& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx120X-all/ `
  "torch==2.11.0+rocm7.13.0" `
  "rocm[devel,libraries]==7.13.0"
& $python -m pip install "triton-windows==3.7.1.post27"

cd C:\path\to\FreeToken-rocm-test
& $python -m pip install --no-build-isolation .
& $python scripts\setup_windows_rocm_gfx1201.py
. "C:\Users\<user>\FreeToken-rocm-finaltest\.venv\Scripts\freetoken-rocm-gfx1201.ps1"
```

The AMD wheel index and torch version follow AMD's ROCm 7.13 gfx120X installation guidance.
The setup command validates the exact supported versions and hardware, creates writable
external caches, applies the deterministic TVM-FFI Windows JIT compatibility patch, and writes
the activation script shown above. Re-running it is safe; a changed or manually edited TVM-FFI
file is rejected by hash.

## Configured process environment

The generated activation script sets:

```text
ROCM_SDK_TARGET_FAMILY=gfx120X-all
PYTORCH_ROCM_ARCH=gfx1201
TRITON_OVERRIDE_ARCH=gfx1201
TVM_FFI_ROCM_ARCH_LIST=gfx1201
TVM_FFI_GPU_BACKEND=hip
HIP_PATH=<active venv>\Lib\site-packages\_rocm_sdk_core
ROCM_HOME=<same SDK>
ROCM_PATH=<same SDK>
HIP_DEVICE_LIB_PATH=<same SDK>\lib\llvm\amdgcn\bitcode
CC=<same SDK>\lib\llvm\bin\clang-cl.exe
CXX=<same SDK>\lib\llvm\bin\clang++.exe
TRITON_CACHE_DIR=<user home>\FreeToken-rocm-cache\triton
TVM_FFI_CACHE_DIR=<user home>\FreeToken-rocm-cache\tvm-ffi
```

Use `--cache-root C:\some\space-free\path` when the default user path contains spaces.

## Qwen3.6 correctness launch

Before loading this model, stop any older FreeToken model workers and make sure Windows has
enough available commit (RAM plus page file) for the pageable expert banks. Windows error 1455
(`The paging file is too small`) during `safetensors.safe_open` is a host commit-limit failure,
not a HIP kernel failure. The setup script does not change page-file settings.

```powershell
$env:FREETOKEN_PIN_BUDGET_GB = "12"
$env:FREETOKEN_FORCE_E4M3_EMU = "1"
Remove-Item Env:FREETOKEN_ROCM_SAFE_OFFLOAD_COPY -ErrorAction SilentlyContinue

ft serve `
  --model "C:\models\Qwen3.6-35B-A3B-NVFP4" `
  --moe-backend offload `
  --nvfp4-backend triton `
  --expert-load serial `
  --num-pages 2048 `
  --cuda-graph-max-bs 0
```

`FREETOKEN_PIN_BUDGET_GB=12` is the validated test value. On the RX 9070 XT run it mapped
approximately 11.854 GiB across 168 banks; the remaining 72 pageable banks used the safe H2D
fallback. Raw Windows host virtual addresses are never passed to GPU-dereference kernels unless
a valid HIP device mapping exists.

## Known limitations

- This is a custom Windows ROCm build, not upstream support.
- TP=1 is validated; multi-rank Windows operation is not.
- CUDA graphs remain disabled for the correctness configuration.
- The partial mapped/pageable offload path is correctness-tested but not performance-tuned.
- The deterministic TVM-FFI patch supports exactly `apache-tvm-ffi==0.1.13.post3`; upgrade only
  after reviewing and updating its source hash and compatibility transformation.
