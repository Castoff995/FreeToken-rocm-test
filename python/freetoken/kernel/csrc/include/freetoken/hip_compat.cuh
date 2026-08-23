#pragma once

// patched: HIP compatibility shim for the CUDA-flavored csrc headers.
// Included only when built with -D__HIP_PLATFORM_AMD__=1 (see utils.cuh).
#include <hip/hip_runtime.h>

#include <tuple>

#ifndef __always_inline
#define __always_inline inline __attribute__((always_inline))
#endif

#ifndef __grid_constant__
#define __grid_constant__  // patched: no-op under HIP
#endif

using cudaError_t = ::hipError_t;
using cudaStream_t = ::hipStream_t;
using cudaLaunchConfig_t = ::hipLaunchConfig_t;
using cudaLaunchAttribute = ::hipLaunchAttribute;

inline constexpr auto cudaSuccess = ::hipSuccess;
inline constexpr auto cudaFuncAttributeMaxDynamicSharedMemorySize =
    ::hipFuncAttributeMaxDynamicSharedMemorySize;

[[nodiscard]] inline auto cudaGetErrorString(::cudaError_t e) -> const char * {
  return ::hipGetErrorString(e);
}

inline auto cudaGetLastError() -> ::cudaError_t { return ::hipGetLastError(); }

// ---- device/host-pointer APIs used by fast_index_copy.cuh ----
inline constexpr auto cudaDevAttrUnifiedAddressing =
    ::hipDeviceAttributeUnifiedAddressing;
inline constexpr auto cudaDevAttrCanUseHostPointerForRegisteredMem =
    ::hipDeviceAttributeCanUseHostPointerForRegisteredMem;

template <typename... A>
inline auto cudaGetDevice(A &&...args) -> ::cudaError_t {
  return ::hipGetDevice(args...);
}
template <typename... A>
inline auto cudaDeviceGetAttribute(A &&...args) -> ::cudaError_t {
  return ::hipDeviceGetAttribute(args...);
}
template <typename... A>
inline auto cudaSetDevice(A &&...args) -> ::cudaError_t {
  return ::hipSetDevice(args...);
}
template <typename... A>
inline auto cudaHostGetDevicePointer(A &&...args) -> ::cudaError_t {
  return ::hipHostGetDevicePointer(args...);
}

template <typename F>
inline auto cudaFuncSetAttribute(F *func, ::hipFuncAttribute attr, int value)
    -> ::cudaError_t {
  return ::hipFuncSetAttribute(reinterpret_cast<const void *>(func), attr,
                               value);
}

// Extended launch: HIP runtime resolves __global__ symbol pointers by address.
template <typename F, typename... Args>
inline auto cudaLaunchKernelEx(const ::cudaLaunchConfig_t *config, F func,
                               Args &&...args) -> ::cudaError_t {
  auto storage = std::make_tuple(args...);
  return [&]<std::size_t... I>(std::index_sequence<I...>) {
    void *params[] = {const_cast<void *>(
        static_cast<const void *>(&std::get<I>(storage)))...};
    return ::hipLaunchKernelExC(config, reinterpret_cast<const void *>(func),
                                params);
  }(std::index_sequence_for<Args...>{});
}
