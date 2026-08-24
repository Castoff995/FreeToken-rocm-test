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

A direct comparison against official llama.cpp was attempted using the same
`Qwen3.6-35B-A3B-NVFP4` source checkpoint.

The checkpoint was converted with the official llama.cpp converter to a
`MOSTLY_NVFP4` GGUF. The NVFP4 expert weights were transferred directly rather
than requantized. The remaining FP8 tensors were represented as BF16 by the
converter.

The resulting configuration was:

```text
llama.cpp=b10566
commit=bb4caa754
backend=HIP
ROCm=7.14.0
GPU=AMD Radeon RX 9070 XT
GPU_arch=gfx1201

model=Qwen3.6-35B-A3B-NVFP4
parameters=34,660,672,370
GGUF_file_type=MOSTLY_NVFP4
GGUF_size=20.638 GiB

API_ready=YES
VRAM_after_load≈14.946 GiB
```

The model loaded successfully, the RX 9070 XT was detected as `gfx1201`, and
the OpenAI-compatible API became available.

However, the first real inference request terminated the llama.cpp backend
during matrix multiplication:

```text
ggml_cuda_compute_forward: MUL_MAT failed
ROCm error: invalid argument
ggml-cuda.cu:2408
```

The failure terminated the llama.cpp process, but did not cause a Windows TDR
or GPU driver reset:

```text
HIP_fatal=YES
TDR=NO
driver_reset=NO
```

Because llama.cpp did not produce a successful first token, no valid TTFT,
decode throughput, translation, or `llama-bench` performance measurements were
collected.

The resulting classification is:

```text
PARITY_CLASS=EXACT_MODEL_NVFP4_GGUF
ENGINE_COMPARISON_VALID=NO
ENGINE_VERDICT=INCONCLUSIVE_LLAMA_CPP_HIP_NVFP4_RUNTIME_FAILURE
```

This should **not** be interpreted as a measured FreeToken performance victory:
there are no comparable llama.cpp throughput numbers. It is instead a practical
availability result for the tested configuration: the custom FreeToken build
completed stable end-to-end inference on this system, while the exact-model
llama.cpp HIP/NVFP4 path failed at its first real compute.

The llama.cpp failure is tracked upstream:

[ggml-org/llama.cpp #27670 — Windows HIP gfx1201: Qwen3.6-35B-A3B NVFP4 loads, then first MUL_MAT fails with ROCm invalid argument](https://github.com/ggml-org/llama.cpp/issues/27670)

## Current status

For the tested Windows / RX 9070 XT / gfx1201 configuration:

```text
BEST_CURRENT_CONFIRMED_E2E_BUILD=CUSTOM_FREETOKEN
BEST_VALIDATED_PIN_BUDGET_GIB=12
CUSTOM_MEDIAN_DECODE=13.92 tok/s
LLAMA_CPP_EXACT_NVFP4_PERFORMANCE=NOT_MEASURABLE
```

A future engine-to-engine performance comparison remains pending either:

- an upstream fix for the llama.cpp HIP/NVFP4 `MUL_MAT` failure; or
- a separate practical comparison using another llama.cpp quantization, with
  the quantization difference stated explicitly.
```
