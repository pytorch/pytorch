// nccl_symm_comm.cuh
// Host-side and device-side classes for NCCL symmetric memory communication
//
// This file contains:
// 1. SymmContext - abstract base class for symmetric communication context
// 2. NCCLSymmContext - NCCL-specific implementation holding ncclWindow and
// devComm
// 3. NCCLSymmComm - Host-side class for NCCL communicator initialization
//
// Based on NCCL documentation for Host-Side setup and Simple LSA Kernel
// patterns.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

// Include NCCL headers for proper type definitions
// When building with NCCL 2.28+, these provide the real types
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

// Mark functions as extern "C" to avoid C++ name mangling for use with Triton
extern "C" {

// =============================================================================
// ABSTRACT SYMMETRIC CONTEXT
// Base class that abstracts away the specific communication backend
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
// Contains NCCL-specific data including ncclWindow and devComm
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
 * This context is created on the host and passed to device kernels.
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

} // extern "C"
