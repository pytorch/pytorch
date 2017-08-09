/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @file hip_runtime_api.h
 *
 * @brief Defines the API signatures for HIP runtime.
 * This file can be compiled with a standard compiler.
 */

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_API_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_API_H


#include <string.h> // for getDeviceProp
#include <hip/hip_common.h>

enum {
HIP_SUCCESS = 0,
HIP_ERROR_INVALID_VALUE,
HIP_ERROR_NOT_INITIALIZED,
HIP_ERROR_LAUNCH_OUT_OF_RESOURCES
};

typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics    : 1;   ///< 32-bit integer atomics for global memory.
    unsigned hasGlobalFloatAtomicExch : 1;   ///< 32-bit float atomic exch for global memory.
    unsigned hasSharedInt32Atomics    : 1;   ///< 32-bit integer atomics for shared memory.
    unsigned hasSharedFloatAtomicExch : 1;   ///< 32-bit float atomic exch for shared memory.
    unsigned hasFloatAtomicAdd        : 1;   ///< 32-bit float atomic add in global and shared memory.

    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics    : 1;   ///< 64-bit integer atomics for global memory.
    unsigned hasSharedInt64Atomics    : 1;   ///< 64-bit integer atomics for shared memory.

    // Doubles
    unsigned hasDoubles               : 1;   ///< Double-precision floating point.

    // Warp cross-lane operations
    unsigned hasWarpVote              : 1;   ///< Warp vote instructions (__any, __all).
    unsigned hasWarpBallot            : 1;   ///< Warp ballot instructions (__ballot).
    unsigned hasWarpShuffle           : 1;   ///< Warp shuffle operations. (__shfl_*).
    unsigned hasFunnelShift           : 1;   ///< Funnel two words into one with shift&mask caps.

    // Sync
    unsigned hasThreadFenceSystem     : 1;   ///< __threadfence_system.
    unsigned hasSyncThreadsExt        : 1;   ///< __syncthreads_count, syncthreads_and, syncthreads_or.

    // Misc
    unsigned hasSurfaceFuncs          : 1;   ///< Surface functions.
    unsigned has3dGrid                : 1;   ///< Grid and group dims are 3D (rather than 2D).
    unsigned hasDynamicParallelism    : 1;   ///< Dynamic parallelism.
} hipDeviceArch_t;


//---
// Common headers for both NVCC and HCC paths:

/**
 * hipDeviceProp
 *
 */
typedef struct hipDeviceProp_t {
    char name[256];                             ///< Device name.
    size_t totalGlobalMem;                      ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;                   ///< Size of shared memory region (in bytes).
    int regsPerBlock;                           ///< Registers per block.
    int warpSize;                               ///< Warp size.
    int maxThreadsPerBlock;                     ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];                       ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];                         ///< Max grid dimensions (XYZ).
    int clockRate;                              ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;                        ///< Max global memory clock frequency in khz.
    int memoryBusWidth;                         ///< Global memory bus width in bits.
    size_t totalConstMem;                       ///< Size of shared memory region (in bytes).
    int major;                                  ///< Major compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps.
    int minor;                                  ///< Minor compute capability.  On HCC, this is an approximation and features may differ from CUDA CC.  See the arch feature flags for portable ways to query feature caps.
    int multiProcessorCount;                    ///< Number of multi-processors (compute units).
    int l2CacheSize;                            ///< L2 cache size.
    int maxThreadsPerMultiProcessor;            ///< Maximum resident threads per multi-processor.
    int computeMode;                            ///< Compute mode.
    int clockInstructionRate;                   ///< Frequency in khz of the timer used by the device-side "clock*" instructions.  New for HIP.
    hipDeviceArch_t arch;                       ///< Architectural feature flags.  New for HIP.
    int concurrentKernels;                      ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;                            ///< PCI Domain ID
    int pciBusID;                               ///< PCI Bus ID.
    int pciDeviceID;                            ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;    ///< Maximum Shared Memory Per Multiprocessor.
    int isMultiGpuBoard;                        ///< 1 if device is on a multi-GPU board, 0 if not.
    int canMapHostMemory;                       ///< Check whether HIP can map host memory
    int gcnArch;                                ///< AMD GCN Arch Value. Eg: 803, 701
 } hipDeviceProp_t;


/**
 * Memory type (for pointer attributes)
 */
enum hipMemoryType {
    hipMemoryTypeHost,   ///< Memory is physically located on host
    hipMemoryTypeDevice  ///< Memory is physically located on device. (see deviceId for specific device)
};



/**
 * Pointer attributes
 */
typedef struct hipPointerAttribute_t {
    enum hipMemoryType memoryType;
    int device;
    void *devicePointer;
    void *hostPointer;
    int isManaged;
    unsigned allocationFlags; /* flags specified when memory was allocated*/
    /* peers? */
} hipPointerAttribute_t;


// hack to get these to show up in Doxygen:
/**
 *     @defgroup GlobalDefs Global enum and defines
 *     @{
 *
 */


/*
 * @brief hipError_t
 * @enum
 * @ingroup Enumerations
 */
// Developer note - when updating these, update the hipErrorName and hipErrorString functions in NVCC and HCC paths
// Also update the hipCUDAErrorTohipError function in NVCC path.

typedef enum hipError_t {
    hipSuccess                      = 0,    ///< Successful completion.
    hipErrorOutOfMemory             = 2,
    hipErrorNotInitialized          = 3,
    hipErrorDeinitialized           = 4,
    hipErrorProfilerDisabled        = 5,
    hipErrorProfilerNotInitialized  = 6,
    hipErrorProfilerAlreadyStarted  = 7,
    hipErrorProfilerAlreadyStopped  = 8,
    hipErrorInsufficientDriver      = 35,
    hipErrorInvalidImage            = 200,
    hipErrorInvalidContext          = 201,  ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent   = 202,
    hipErrorMapFailed               = 205,
    hipErrorUnmapFailed             = 206,
    hipErrorArrayIsMapped           = 207,
    hipErrorAlreadyMapped           = 208,
    hipErrorNoBinaryForGpu          = 209,
    hipErrorAlreadyAcquired         = 210,
    hipErrorNotMapped               = 211,
    hipErrorNotMappedAsArray        = 212,
    hipErrorNotMappedAsPointer      = 213,
    hipErrorECCNotCorrectable       = 214,
    hipErrorUnsupportedLimit        = 215,
    hipErrorContextAlreadyInUse     = 216,
    hipErrorPeerAccessUnsupported   = 217,
    hipErrorInvalidKernelFile       = 218,  ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext  = 219,
    hipErrorInvalidSource           = 300,
    hipErrorFileNotFound            = 301,
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed  = 303,
    hipErrorOperatingSystem         = 304,
    hipErrorSetOnActiveProcess      = 305,
    hipErrorInvalidHandle           = 400,
    hipErrorNotFound                = 500,
    hipErrorIllegalAddress          = 700,
    hipErrorInvalidSymbol           = 701,
// Runtime Error Codes start here.
    hipErrorMissingConfiguration    = 1001,
    hipErrorMemoryAllocation        = 1002,    ///< Memory allocation error.
    hipErrorInitializationError     = 1003,    ///< TODO comment from hipErrorInitializationError
    hipErrorLaunchFailure           = 1004,    ///< An exception occurred on the device while executing a kernel.
    hipErrorPriorLaunchFailure      = 1005,
    hipErrorLaunchTimeOut           = 1006,
    hipErrorLaunchOutOfResources    = 1007,    ///< Out of resources error.
    hipErrorInvalidDeviceFunction   = 1008,
    hipErrorInvalidConfiguration    = 1009,
    hipErrorInvalidDevice           = 1010,   ///< DeviceID must be in range 0...#compute-devices.
    hipErrorInvalidValue            = 1011,   ///< One or more of the parameters passed to the API call is NULL or not in an acceptable range.
    hipErrorInvalidDevicePointer    = 1017,   ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection  = 1021,   ///< Invalid memory copy direction
    hipErrorUnknown                 = 1030,   ///< Unknown error.
    hipErrorInvalidResourceHandle   = 1033,   ///< Resource handle (hipEvent_t or hipStream_t) invalid.
    hipErrorNotReady                = 1034,   ///< Indicates that asynchronous operations enqueued earlier are not ready.  This is not actually an error, but is used to distinguish from hipSuccess (which indicates completion).  APIs that return this error include hipEventQuery and hipStreamQuery.
    hipErrorNoDevice                = 1038,   ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorPeerAccessAlreadyEnabled = 1050,  ///< Peer access was already enabled from the current device.

    hipErrorPeerAccessNotEnabled    = 1051,   ///< Peer access was never enabled from the current device.
    hipErrorRuntimeMemory           = 1052,                  ///< HSA runtime memory call returned error.  Typically not seen in production systems.
    hipErrorRuntimeOther            = 1053,                   ///< HSA runtime call other than memory returned error.  Typically not seen in production systems.
    hipErrorHostMemoryAlreadyRegistered = 1061, ///< Produced when trying to lock a page-locked memory.
    hipErrorHostMemoryNotRegistered = 1062,   ///< Produced when trying to unlock a non-page-locked memory.
    hipErrorMapBufferObjectFailed = 1071,   ///< Produced when the IPC memory attach failed from ROCr.
    hipErrorTbd                             ///< Marker that more error codes are needed.
} hipError_t;

/*
 * @brief hipDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum hipDeviceAttribute_t {
    hipDeviceAttributeMaxThreadsPerBlock,                   ///< Maximum number of threads per block.
    hipDeviceAttributeMaxBlockDimX,                         ///< Maximum x-dimension of a block.
    hipDeviceAttributeMaxBlockDimY,                         ///< Maximum y-dimension of a block.
    hipDeviceAttributeMaxBlockDimZ,                         ///< Maximum z-dimension of a block.
    hipDeviceAttributeMaxGridDimX,                          ///< Maximum x-dimension of a grid.
    hipDeviceAttributeMaxGridDimY,                          ///< Maximum y-dimension of a grid.
    hipDeviceAttributeMaxGridDimZ,                          ///< Maximum z-dimension of a grid.
    hipDeviceAttributeMaxSharedMemoryPerBlock,              ///< Maximum shared memory available per block in bytes.
    hipDeviceAttributeTotalConstantMemory,                  ///< Constant memory size in bytes.
    hipDeviceAttributeWarpSize,                             ///< Warp size in threads.
    hipDeviceAttributeMaxRegistersPerBlock,                 ///< Maximum number of 32-bit registers available to a thread block. This number is shared by all thread blocks simultaneously resident on a multiprocessor.
    hipDeviceAttributeClockRate,                            ///< Peak clock frequency in kilohertz.
    hipDeviceAttributeMemoryClockRate,                      ///< Peak memory clock frequency in kilohertz.
    hipDeviceAttributeMemoryBusWidth,                       ///< Global memory bus width in bits.
    hipDeviceAttributeMultiprocessorCount,                  ///< Number of multiprocessors on the device.
    hipDeviceAttributeComputeMode,                          ///< Compute mode that device is currently in.
    hipDeviceAttributeL2CacheSize,                          ///< Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
    hipDeviceAttributeMaxThreadsPerMultiProcessor,          ///< Maximum resident threads per multiprocessor.
    hipDeviceAttributeComputeCapabilityMajor,               ///< Major compute capability version number.
    hipDeviceAttributeComputeCapabilityMinor,               ///< Minor compute capability version number.
    hipDeviceAttributeConcurrentKernels,                    ///< Device can possibly execute multiple kernels concurrently.
    hipDeviceAttributePciBusId,                             ///< PCI Bus ID.
    hipDeviceAttributePciDeviceId,                          ///< PCI Device ID.
    hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,     ///< Maximum Shared Memory Per Multiprocessor.
    hipDeviceAttributeIsMultiGpuBoard,                      ///< Multiple GPU devices.
} hipDeviceAttribute_t;


/**
 *     @}
 */

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
#include "hip/hcc_detail/hip_runtime_api.h"
#elif defined(__HIP_PLATFORM_NVCC__) && !defined (__HIP_PLATFORM_HCC__)
#include "hip/nvcc_detail/hip_runtime_api.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_HCC__ or __HIP_PLATFORM_NVCC__");
#endif


/**
 * @brief: C++ wrapper for hipMalloc
 *
 * Perform automatic type conversion to eliminate need for excessive typecasting (ie void**)
 *
 * @see hipMalloc
 */
#ifdef __cplusplus
template<class T>
static inline hipError_t hipMalloc ( T** devPtr, size_t size)
{
    return hipMalloc((void**)devPtr, size);
}

// Provide an override to automatically typecast the pointer type from void**, and also provide a default for the flags.
template<class T>
static inline hipError_t hipHostMalloc( T** ptr, size_t size, unsigned int flags = hipHostMallocDefault)
{
    return hipHostMalloc((void**)ptr, size, flags);
}
#endif

#endif
