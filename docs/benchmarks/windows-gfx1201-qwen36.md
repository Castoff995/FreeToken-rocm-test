# Windows ROCm gfx1201 — Qwen3.6 validation and benchmark

This document records empirical results from a custom FreeToken build. It does
not represent official support or validation by FlashML or AMD. Results apply
to the specific Windows/gfx1201 system and software versions listed below.

## Tested custom stack

| Component | Configuration |
| --- | --- |
| Operating system | Windows |
| GPU | AMD Radeon RX 9070 XT |
| GPU architecture | gfx1201 |
| GPU driver | 32.0.31035.1003 |
| System RAM | 31.617 GiB |
| Pagefile | `D:\pagefile.sys` |
| Pagefile allocation | 64 GiB allocated, 96 GiB maximum |
| Commit limit | approximately 95.62 GiB |
| Python | 3.12 |
| PyTorch | 2.11.0+rocm7.13.0 |
| HIP | 7.13.99004 |
| ROCm | official 7.13 gfx120X-all |
| Tensor parallelism | TP=1 |
| Model | Qwen3.6 |

The custom milestone was validated at commit
`5924dad2513160f663542ed005bcb22bfbea3cbd` (`feat(rocm): support Windows
gfx1201 Qwen3.6 serving`) and is recorded as:

```text
MILESTONE=WINDOWS_ROCM_GFX1201_QWEN36_E2E_REPRODUCIBLE
```

## Reproducible end-to-end result

```text
expert_source_load=PASS
pin_budget=PASS
warmup_embed=PASS
layer0=PASS
API_ready=PASS
HTTP=200
first_token=PASS
decode_started=PASS
backend_alive=PASS
TDR=NO
driver_reset=NO
HIP_fatal=NO
```

One example response was:

```text
2 plus 3 equals 5.
```

A three-request stability smoke test also passed, including a Chinese-to-Russian
translation request. This is evidence of repeatable operation on the tested
machine, not broad production validation.

## 12 GiB pin-budget baseline

The stable baseline used `FREETOKEN_PIN_BUDGET_GB=12`.

| Measurement | Result |
| --- | ---: |
| Successfully pinned | 11.854 GiB |
| Skipped because of budget | 5.080 GiB |
| Mapped banks | 168 |
| Pageable banks | 72 |
| Mapped fast-path layers | 28/40 |
| Safe-copy layers | 12/40 |
| Host-registration failures | 0 |
| RAM used after load | 25.540 GiB |
| Commit used after load | 49.126 GiB |
| VRAM used after load | approximately 14.151 GiB |
| Model load time | approximately 60.5 s |
| API ready time | approximately 71.5 s |

The 28/40 fast-path and 12/40 safe-copy split does **not** describe an
abridged model. All 40 layers are used. The split only describes whether each
layer's expert banks use mapped/pinned host access or pageable host memory with
the safe PyTorch host-to-device copy path.

### Synthetic repeats

Each repeat returned HTTP 200 with 255 completion tokens and
`finish_reason=length`.

| Repeat | TTFT | Decode | End-to-end |
| --- | ---: | ---: | ---: |
| 1 | 8446.132 ms | 13.621 tok/s | 27.095 s |
| 2 | 1187.586 ms | 13.975 tok/s | 19.364 s |
| 3 | 1151.169 ms | 13.920 tok/s | 19.399 s |

Repeat 1 had a substantially larger time to first token, so the median is used
for the summary:

```text
median_TTFT=1187.6 ms
median_decode=13.92 tok/s
```

### Translation benchmark

```text
prompt_tokens=53
completion_tokens=47
TTFT=1058.731 ms
E2E=4.023652 s
average_completion_tps=11.681
finish_reason=stop
```

Output:

```text
Наступила весна, расцвели цветы, птицы поют на деревьях. Далекие горы окутаны лёгкой дымкой, а утренний воздух наполнен ароматом влажной земли.
```

## 14 GiB boundary test

| Measurement | Result |
| --- | ---: |
| Requested pin budget | 14 GiB |
| Attempted pinning | 13.970 GiB |
| Successfully pinned | 13.845 GiB |
| Skipped because of budget | 2.963 GiB |
| Mapped banks | 197 |
| Pageable banks | 43 |
| Mapped fast-path layers | 32/40 |
| Safe-copy layers | 8/40 |
| Host-registration failures | 1 |
| RAM used after load | 25.760 GiB |
| Commit used after load | 48.863 GiB |
| Model load time | approximately 47.5 s |
| API ready | NO |
| Chat endpoint | HTTP 503: model is still loading |
| TDR | NO |
| Driver reset | NO |
| Fatal HIP error | NO |

The first registration failure was:

```text
hipHostRegister(134217728 bytes) failed with hipError 1
```

The backend did not complete prefill warmup or reach API readiness after more
than five minutes. The sweep was stopped according to the safety rules; 16 GiB
and 17 GiB were not tested.

The resulting recommendation is:

```text
BEST_STABLE_PIN_BUDGET_GIB=12
SAFE_RECOMMENDED_DAILY_BUDGET=12
MAX_TESTED_STABLE_BUDGET=12
FULL_PINNING_POSSIBLE=NOT_DEMONSTRATED
Windows_host_registration_limit_hit=YES at the 14 GiB test
```

Twelve GiB is the best validated stable value in this sweep. It is not claimed
to be an absolute Windows host-registration maximum.

## Upstream comparison

### Pure upstream main

The clean `FlashML-org/FreeToken` `main` checkout was tested at
`bd372b630a028e3faa51f4ab0ef6a98c2f2de501`. It did not reach API readiness.
The first blocker was:

```text
RuntimeError: CUDA_HOME is required to build
freetoken.kernel._pinned_tensor because it links against
the CUDA runtime API.
```

```text
classification=PURE_UPSTREAM_MAIN_NOT_ROCM_READY
```

No performance values are reported because serving did not start.

### Upstream PR #137

The clean `FlashML-org/FreeToken` PR #137 checkout was tested at
`7e1d7421b96feaead8398455283cb9e73c091181`. It did not reach API readiness.
The first blocker was:

```text
pinned_tensor.obj : error LNK2001:
unresolved external symbol
c10::ValueError::ValueError(c10::SourceLocation, std::string)

_pinned_tensor.cp312-win_amd64.pyd :
fatal error LNK1120: 1 unresolved externals
```

```text
classification=UPSTREAM_PR137_NOT_E2E_READY
```

PR #132 was consulted only as a reference. PR #132 and PR #137 were not
combined, and no custom fixes were injected into either upstream checkout.

Among the configurations tested on this Windows/gfx1201 machine, only the
custom build was confirmed end-to-end at the time of testing:

```text
BEST_CURRENT_BUILD=CUSTOM
CUSTOM_12G_OR_UPSTREAM=CUSTOM_12G
```

Да. Я бы **заменил всё начиная с `## Comparison summary` и до конца** вот на это:

```md
## Comparison summary

| Metric | Custom 12 GiB | Upstream main | PR #137 | llama.cpp HIP / exact NVFP4 |
| --- | ---: | ---: | ---: | ---: |
| API ready | YES | NO | NO | YES |
| Model / quantization | Qwen3.6 NVFP4 | N/A | N/A | Same Qwen3.6 NVFP4 checkpoint |
| Pinned GiB | 11.854 | N/A | N/A | N/A |
| Fast layers | 28/40 | N/A | N/A | N/A |
| Safe-copy layers | 12/40 | N/A | N/A | N/A |
| VRAM after load | approximately 14.151 GiB | N/A | N/A | approximately 14.946 GiB |
| Median TTFT | 1187.6 ms | N/A | N/A | N/A |
| Median decode | 13.92 tok/s | N/A | N/A | N/A |
| Translation E2E | 4.024 s | N/A | N/A | N/A |
| E2E serving | PASS | Not runnable | Not runnable | FAIL on first inference |
| TDR / reset | NO / NO | NO / NO | NO / NO | NO / NO |

## llama.cpp exact-NVFP4 comparison

llama.cpp=b10566
commit=bb4caa754
ROCm=7.14.60850
GPU=RX 9070 XT / gfx1201
ROCBLAS_USE_HIPBLASLT=0

API_READY=YES
FIRST_INFERENCE=PASS
HIP_FATAL=NO
TDR=NO
driver_reset=NO

R1_DECODE=14.618 tok/s
R2_DECODE=15.954 tok/s
R3_DECODE=15.918 tok/s

MEDIAN_DECODE=15.918 tok/s
MEDIAN_TTFT=177.303 ms

FREETOKEN_MEDIAN_DECODE=13.920 tok/s

ABSOLUTE_DIFFERENCE=+1.998 tok/s
PERCENT_DIFFERENCE=+14.353%
PERFORMANCE_WINNER=LLAMA_CPP

This comparison applies only to the tested Qwen3.6 model,
RX 9070 XT, Windows and software configurations.

llama.cpp requires ROCBLAS_USE_HIPBLASLT=0 because the default
ROCm 7.14 hipBLASLt path is affected by an upstream gfx1201 bug.

## Current status

For the tested Windows / RX 9070 XT / gfx1201 configuration:

```text
BEST_VALIDATED_FREETOKEN_PIN_BUDGET_GIB=12
FREETOKEN_MEDIAN_DECODE=13.920 tok/s

LLAMA_CPP_WORKAROUND=ROCBLAS_USE_HIPBLASLT=0
LLAMA_CPP_API_READY=YES
LLAMA_CPP_FIRST_INFERENCE=PASS
LLAMA_CPP_MEDIAN_DECODE=15.918 tok/s
LLAMA_CPP_MEDIAN_TTFT=177.303 ms

ABSOLUTE_DIFFERENCE=+1.998 tok/s
PERCENT_DIFFERENCE=+14.353%
PERFORMANCE_WINNER=LLAMA_CPP

HIP_FATAL=NO
TDR=NO
DRIVER_RESET=NO
For this specific Qwen3.6 model, RX 9070 XT, Windows, and tested software
configuration, llama.cpp with ROCBLAS_USE_HIPBLASLT=0 achieved approximately
14.35% higher median decode throughput than the validated custom FreeToken
12 GiB configuration.
This is not a general engine-level performance conclusion.
The default ROCm 7.14 hipBLASLt routing path on Windows/gfx1201 remains affected
by an upstream sequence/state-dependent GEMM bug. The llama.cpp result above
therefore depends on the ROCBLAS_USE_HIPBLASLT=0 workaround.
Relevant upstream tracking:
- llama.cpp: https://github.com/ggml-org/llama.cpp/issues/27670
- ROCm: https://github.com/ROCm/legacy-rocm-build/issues/6461
