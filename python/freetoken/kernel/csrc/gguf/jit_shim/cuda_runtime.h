#pragma once

// Windows/ROCm JIT shim: minimal CUDA runtime API compatibility over HIP.
// TheRock's HIP SDK ships no cuda_runtime.h, but torch's c10/cuda headers
// unconditionally include it. Types are aliased and functions #define-renamed
// to their hip* counterparts (same signatures), so no new symbols are needed.

#include <hip/hip_runtime.h>

#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

// ---- types ----
typedef hipError_t cudaError_t;
typedef hipStream_t cudaStream_t;
typedef hipEvent_t cudaEvent_t;
typedef hipMemcpyKind cudaMemcpyKind;
typedef hipDeviceProp_t cudaDeviceProp;
typedef hipPointerAttribute_t cudaPointerAttributes;
typedef hipStreamCaptureMode cudaStreamCaptureMode;
typedef hipStreamCaptureStatus cudaStreamCaptureStatus;
typedef hipGraph_t cudaGraph_t;
typedef hipGraphExec_t cudaGraphExec_t;
typedef hipHostFn_t cudaHostFn_t;
typedef hipUserObject_t cudaUserObject_t;

// ---- enum constants ----
#define cudaSuccess hipSuccess
#define cudaErrorNotReady hipErrorNotReady
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaEventDefault hipEventDefault
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamDefault hipStreamDefault
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureStatusInvalidated hipStreamCaptureStatusInvalidated
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#ifndef cudaMemoryTypeUnregistered
#define cudaMemoryTypeUnregistered hipMemoryTypeUnregistered
#endif

// ---- functions ----
#define cudaGetDevice hipGetDevice
#define cudaSetDevice hipSetDevice
#define cudaGetLastError hipGetLastError
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithPriority hipStreamCreateWithPriority
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamQuery hipStreamQuery
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo
#define cudaStreamGetCaptureInfo_v2 hipStreamGetCaptureInfo_v2
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaThreadExchangeStreamCaptureMode hipThreadExchangeStreamCaptureMode
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventRecordWithFlags hipEventRecordWithFlags
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaFreeHost hipHostFree
#define cudaMallocAsync hipMallocAsync
#define cudaFreeAsync hipFreeAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyAsyncPeer hipMemcpyPeerAsync
#define cudaMemsetAsync hipMemsetAsync
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaStreamGetPriority hipStreamGetPriority
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphInstantiateWithFlags hipGraphInstantiateWithFlags
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaUserObjectCreate hipUserObjectCreate
#define cudaUserObjectRelease hipUserObjectRelease
#define cudaUserObjectRetain hipUserObjectRetain
#define cudaUserObjectNoDestructorSync hipUserObjectNoDestructorSync
#define cudaGraphUserObjectMove hipGraphUserObjectMove
#define cudaGraphRetainUserObject hipGraphRetainUserObject

// ---- warp sync builtins ----
// ROCm's __*_sync templates (amd_warp_sync_functions.h) static_assert 64-bit
// masks, which vendored CUDA sources (32-bit 0xffffffff literals) fail. Same
// workaround as llama.cpp's vendors/hip.h: map onto the raw builtins; the mask
// is unused on AMD.
#define __shfl_sync(mask, var, srcLane, ...) __shfl((var), (srcLane), ##__VA_ARGS__)
#define __shfl_up_sync(mask, var, delta, ...) __shfl_up((var), (delta), ##__VA_ARGS__)
#define __shfl_down_sync(mask, var, delta, ...) __shfl_down((var), (delta), ##__VA_ARGS__)
#define __shfl_xor_sync(mask, var, laneMask, ...) __shfl_xor((var), (laneMask), ##__VA_ARGS__)
#define __syncwarp(mask)
