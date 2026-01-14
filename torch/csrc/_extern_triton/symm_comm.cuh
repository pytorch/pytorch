// symm_comm.cuh
// Unified header for symmetric memory communication contexts
//
// This file provides:
// 1. SymmContext - Abstract base class for symmetric communication
// 2. NCCLSymmContext - NCCL-specific context (uses NCCL LSA API)
// 3. NVSHMEMSymmContext - NVSHMEM-specific context (uses NVSHMEM API)
//
// The unified design allows kernels to work with either backend through
// runtime type dispatch.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// =============================================================================
// NCCL TYPE DEFINITIONS
// =============================================================================

// Include NCCL headers for proper type definitions when available
#if __has_include(<nccl.h>)
#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include <nccl_device.h>
#define NCCL_SYMM_TYPES_AVAILABLE
#endif
#endif

// Fallback forward declarations only if NCCL types are not available
#ifndef NCCL_SYMM_TYPES_AVAILABLE
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ncclComm* ncclComm_t;
typedef void* ncclWindow_t;
typedef struct ncclDevComm ncclDevComm;
#ifdef __cplusplus
}
#endif
#endif

// =============================================================================
// NVSHMEM DEVICE FUNCTION DECLARATIONS
// These are provided by libnvshmem_device.bc and linked at compile time
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__
// Get pointer to symmetric memory on a remote PE
extern __device__ void* nvshmem_ptr(const void* dest, int pe);

// Barrier synchronization across all PEs (block-level)
extern __device__ void nvshmemx_barrier_all_block();

// Get the calling PE's number
extern __device__ int nvshmem_my_pe();

// Get the number of PEs
extern __device__ int nvshmem_n_pes();

// Float put (single element)
extern __device__ void nvshmem_float_p(float* dest, float value, int pe);

// Float get (single element)
extern __device__ float nvshmem_float_g(const float* source, int pe);

// Block-level put (for larger transfers)
extern __device__ void nvshmemx_float_put_block(
    float* dest,
    const float* source,
    size_t nelems,
    int pe);

// Block-level get (for larger transfers)
extern __device__ void nvshmemx_float_get_block(
    float* dest,
    const float* source,
    size_t nelems,
    int pe);

// Quiet - ensures all prior NVSHMEM operations are complete
extern __device__ void nvshmem_quiet();

// Fence - ensures ordering of memory operations
extern __device__ void nvshmem_fence();
#endif

#ifdef __cplusplus
}
#endif

// =============================================================================
// NCCL DEVICE FUNCTION DECLARATIONS
// Note: NCCL does NOT provide a device bitcode library (unlike NVSHMEM)
// These declarations are here for completeness but will result in unresolved
// symbols when compiling to bitcode for Triton.
// =============================================================================

#ifdef __CUDA_ARCH__
extern "C" __device__ void* ncclGetLsaPointer(
    void* window,
    size_t offset,
    int peer);
extern "C" __device__ void ncclLsaBarrier(void* devComm, int barrierId);
#endif

// Mark structures as extern "C" to avoid C++ name mangling for use with Triton
extern "C" {

// =============================================================================
// SYMMETRIC CONTEXT BASE CLASS
// =============================================================================

/**
 * Abstract base class for symmetric communication context.
 * This allows kernels to work with different backends (NCCL, NVSHMEM, etc.)
 * without knowing the specific implementation details.
 */
struct SymmContext {
  // Type identifier for runtime type checking
  enum class Type : int32_t {
    NCCL = 0,
    NVSHMEM = 1,
  };

  Type type;
  int32_t rank;
  int32_t world_size;

  __host__ __device__ SymmContext(Type t, int32_t r, int32_t ws)
      : type(t), rank(r), world_size(ws) {}
};

// =============================================================================
// NCCL-SPECIFIC SYMMETRIC CONTEXT
// =============================================================================

/**
 * NCCL-specific symmetric communication context.
 *
 * This structure holds all NCCL-related data needed for LSA (Local Symmetric
 * Access) operations including:
 * - ncclWindow_t for the registered symmetric memory buffer
 * - ncclWindow_t for the signal pad (for synchronization)
 * - Pointer to the device communicator
 *
 * LIMITATION: NCCL does not provide a device bitcode library, so this context
 * cannot be used with Triton extern_libs linking. It's included here for
 * completeness and potential future support.
 */
struct NCCLSymmContext : public SymmContext {
  // ncclWindow for the data buffer (registered with ncclCommWindowRegister)
  ncclWindow_t buffer_window;

  // ncclWindow for the signal pad (for put-with-signal operations)
  ncclWindow_t signal_window;

  // Pointer to the NCCL device communicator (created via ncclDevCommCreate)
  ncclDevComm* dev_comm;

  // Local buffer pointer (the base address of symmetric memory)
  void* local_buffer;

  // Size of the buffer in bytes
  size_t buffer_size;

  // Device index where this context is valid
  int32_t device_idx;

  __host__ __device__ NCCLSymmContext()
      : SymmContext(Type::NCCL, 0, 0),
        buffer_window(nullptr),
        signal_window(nullptr),
        dev_comm(nullptr),
        local_buffer(nullptr),
        buffer_size(0),
        device_idx(-1) {}

  __host__ __device__ NCCLSymmContext(
      int32_t rank,
      int32_t world_size,
      ncclWindow_t buf_win,
      ncclWindow_t sig_win,
      ncclDevComm* dcomm,
      void* buf,
      size_t size,
      int32_t dev_idx)
      : SymmContext(Type::NCCL, rank, world_size),
        buffer_window(buf_win),
        signal_window(sig_win),
        dev_comm(dcomm),
        local_buffer(buf),
        buffer_size(size),
        device_idx(dev_idx) {}
};

// =============================================================================
// NVSHMEM-SPECIFIC SYMMETRIC CONTEXT
// =============================================================================

/**
 * NVSHMEM-specific symmetric communication context.
 *
 * This structure holds all NVSHMEM-related data needed for symmetric memory
 * operations. Unlike NCCL, NVSHMEM uses a simpler programming model where
 * symmetric memory is accessed directly via nvshmem_ptr().
 *
 * IMPORTANT: NVSHMEM uses GLOBAL PE numbers (from nvshmem_my_pe()), not
 * group-local ranks. The kernel should use nvshmem_my_pe() and nvshmem_n_pes()
 * for PE numbering, or use the global_rank/global_world_size stored here.
 *
 * Key differences from NCCL:
 * - No window registration needed (NVSHMEM handles this internally)
 * - nvshmem_ptr() returns direct pointers to peer memory
 * - Barrier operations use nvshmemx_barrier_all_block()
 * - PE numbers are GLOBAL, not group-local
 *
 * This context is created on the host and passed to device kernels.
 * NVSHMEM provides libnvshmem_device.bc which can be linked with Triton
 * kernels, making this implementation fully functional.
 */
struct NVSHMEMSymmContext : public SymmContext {
  // Local buffer pointer (the base address of this PE's symmetric memory)
  // This is the symmetric address that can be used with nvshmem_ptr()
  void* local_buffer;

  // Size of the buffer in bytes
  size_t buffer_size;

  // Device index where this context is valid
  int32_t device_idx;

  // Offset within the allocation (for sub-tensors)
  size_t offset;

  // Global PE number (from nvshmem_my_pe(), not group-local rank)
  // Use this with nvshmem_ptr() for remote access
  int32_t global_rank;

  // Total number of PEs in the NVSHMEM job (from nvshmem_n_pes())
  int32_t global_world_size;

  __host__ __device__ NVSHMEMSymmContext()
      : SymmContext(Type::NVSHMEM, 0, 0),
        local_buffer(nullptr),
        buffer_size(0),
        device_idx(-1),
        offset(0),
        global_rank(-1),
        global_world_size(0) {}

  __host__ __device__ NVSHMEMSymmContext(
      int32_t rank,
      int32_t world_size,
      void* local_buf,
      size_t size,
      int32_t dev_idx,
      size_t off,
      int32_t g_rank,
      int32_t g_world_size)
      : SymmContext(Type::NVSHMEM, rank, world_size),
        local_buffer(local_buf),
        buffer_size(size),
        device_idx(dev_idx),
        offset(off),
        global_rank(g_rank),
        global_world_size(g_world_size) {}
};

} // extern "C"
