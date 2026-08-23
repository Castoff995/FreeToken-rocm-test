# Failure Diagnosis Rule — ROCm/Windows Bring-Up

Run this BEFORE attempting any fix. Classify first, fix second.

## Step 0 — Preliminary (always)

1. Kill zombies: any `python`/`ft` process older than the current attempt
   holds log handles → new run writes nothing. Kill, verify logs deleted.
2. Verify env recipe present: `HIP_PATH`, `TVM_FFI_ROCM_ARCH_LIST`,
   `TRITON_OVERRIDE_ARCH`, `CC`; vcvars environment for linking.
3. Confirm log file `LastWriteTime` is fresh before reading it.
   Stale timestamp = process never started = go to zombie rule, do not
   parse old content.

## Truth Table — Stage Classification

| Log evidence (regex)                                    | Stage    | True? → cause                          |
|---------------------------------------------------------|----------|----------------------------------------|
| no new log / stale `LastWriteTime`                      | launch   | zombie process or cmd script died early |
| `clang: error: no such file or directory: '/...'`       | compile  | MSVC-style flags fed to clang           |
| `clang: error: unsupported option '-fPIC' ... windows-msvc` | compile | -fPIC invalid on msvc triple          |
| command line contains `gfx906` (not gfx1201)            | compile  | `TVM_FFI_ROCM_ARCH_LIST` unset          |
| `error: unknown type name '__always_inline'`            | compile  | CUDA/MSVC source idiom, HIP shim needed |
| `error: 'sys/cdefs.h' file not found`                   | compile  | POSIX header unguarded                  |
| `no member named 'programmaticStreamSerialization'`     | compile  | ROCm headers lack PDL attr → gate off   |
| `error: use of undeclared identifier '_InlineInterlockedAdd64'` or `'__assume'` | compile | fake `-D_MSC_VER` on gnu target broke MSVC-branch headers → revert to msvc triple |
| LNK2019/LNK2001 + `LNK1120`                             | link     | missing library — classify by symbol    |
| `Backend worker is gone and cannot be restarted`        | cascade  | downstream symptom of build failure; ignore, fix root |
| `ready to serve on 127.0.0.1:1919`                      | success  | proceed to runtime tests                |

## Link-Symbol Truth Rules

| Symbol pattern in LNK2019                    | Missing lib      | Rule |
|----------------------------------------------|------------------|------|
| starts with `hip`                            | `amdhip64.lib`   | T    |
| starts `__imp_`                              | system lib (kernel32/user32/shell32/advapi32) | T |
| `operator new/delete`, `type_info::vftable`  | C++ stdlib (`libcpmt.lib`) | T |
| `_tls_index`, `_Init_thread_*`, `atexit`, `__chkstk` | CRT (`libcmt.lib`) | T |
| `wcsnlen`, `___lc_*`, `__acrt_iob_func`      | UCRT (`libucrt.lib`) | T |
| `__CxxFrameHandler3/4`, `__GSHandlerCheck*`, `__guard_dispatch_icall_fptr` | `libvcruntime.lib` | T |

If >50% unresolved are plain CRT/system symbols → default-lib chain absent →
link ALL rows explicitly instead of chasing one at a time.

## Flowchart

```
START
  │
  ├─ Fresh log? ──NO──▶ kill zombies, relaunch, recheck ──┐
  │ YES                                                   │
  ├─ "ready to serve"? ──YES──▶ DONE (POST chat sample)   │
  │ NO                                                    │
  ├─ LNK2019/LNK2001/LNK1120? ──YES──▶ link-symbol table │
  │ NO                                ▼ add libs, patch   │
  ├─ "error:" + .cu/.h line?              extension.py    │
  │     └─ flag rejected?  → fix flags in tvm_ffi branch  │
  │     └─ unknown type/header? → shim/guard csrc header  │
  │ NO                                                    │
  ├─ Backend worker gone? ──YES──▶ ignore (cascade), read │
  │                                   earlier lines       │
  ▼                                                       │
Apply ONE root-cause fix ──── relaunch ◀─────────────────-┘
```

## Meta-Rules (True/False)

- T: Fix root cause once where all callers route through it (e.g. flags in
  one patched branch), never patch each call site.
- T: One change per relaunch cycle — otherwise attribution is lost.
- T: Cascade errors (`worker is gone`) are symptoms; scroll up for the real
  first error.
- F: Assume the newest log line is the root cause — it is usually the last
  of many.
- F: Mix static/dynamic CRT guesses blindly — classify by symbol table above.
- F: Reuse a recipe from another project without checking its assumptions
  (`-D_MSC_VER=1900` gnu-target trick breaks tvm-ffi's own MSVC branches).

Applies to: FreeToken Windows/ROCm bring-up (ft serve, tvm-ffi JIT, hipcc,
ninja) — G:\FreeToken.


## Runtime-Stage Addendum (post-build failures)

| Log evidence | Stage | True cause |
|---|---|---|
| \CUDA error: invalid device function\ at first kernel launch | runtime | code object built for wrong arch — check the JIT command line for \--offload-arch=gfx1201\ (T) / env var read but never wired into flags (F) |
| \KeyError: 'Keyword argument X was specified but unrecognised'\ (triton jit.py) | runtime | kwarg name absent from BOTH kernel sig AND backend Options class. Value irrelevant — presence alone fails. NVIDIA-only launch opts (\launch_pdl\) break on AMD backend |

True/False rules learned:
- F: 'kwarg gated false means harmless' — a keyword argument is validated by NAME at bind time; passing \launch_pdl=False\ still crashes a backend lacking the option.
- T: Fix at the shared choke point — add the missing field to the AMD backend Options (accept+ignore) rather than conditioning every call site.
- T: Supervisor logs swallow tracebacks — match the error text format against framework source (grep site-packages for the message string) to find the raiser.
