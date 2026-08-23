#pragma once

// Windows/ROCm JIT shim: torch ROCm wheels ship c10/hip/impl/hip_cmake_macros.h
// but omit the c10/cuda variant that c10/cuda/CUDAMacros.h includes.
#define C10_CUDA_BUILD_SHARED_LIBS
