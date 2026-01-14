// symm_all_reduce.cu
// Unified CUDA device functions for symmetric memory all-reduce operations
//
// This file contains device functions that can be compiled to LLVM bitcode
// (.bc) and linked with Triton kernels via extern_libs.
//
// The implementation provides a unified frontend function that dispatches
// to either NCCL or NVSHMEM backends based on the SymmContext type.
//
// BACKEND SUPPORT:
// - NVSHMEM: Fully functional. NVSHMEM provides libnvshmem_device.bc that
//   can be linked with this bitcode.
// - NCCL: NOT functional. NCCL does not provide a device bitcode library.
//   The NCCL implementation is included for completeness and future support.
//
// Compile to bitcode:
//   clang++ -x cuda --cuda-device-only -emit-llvm -c symm_all_reduce.cu \
//           -o symm_all_reduce.bc --cuda-gpu-arch=sm_80 -O3 \
//           -fcuda-flush-denormals-to-zero

#include <cuda_runtime.h>
#include <stdint.h>

// Include the unified context definitions
#include "symm_comm.cuh"

// Mark functions as extern "C" to avoid C++ name mangling
extern "C" {

// =============================================================================
// NCCL BACKEND IMPLEMENTATION
// Note: NCCL does not provide a device bitcode library (libnccl_device.bc).
// This section is compiled only when NCCL_HAS_DEVICE_BITCODE is defined.
// =============================================================================

// Set to 0 because NCCL does not ship a device bitcode library
#define NCCL_HAS_DEVICE_BITCODE 0

#if NCCL_HAS_DEVICE_BITCODE

/**
 * Cast SymmContext to NCCLSymmContext after runtime type check.
 * Returns nullptr if the context is not NCCL type.
 */
__device__ __forceinline__ NCCLSymmContext* cast_to_nccl_context(
    SymmContext* ctx) {
  if (ctx == nullptr || ctx->type != SymmContext::Type::NCCL) {
    return nullptr;
  }
  return static_cast<NCCLSymmContext*>(ctx);
}

/**
 * Get a pointer to a peer's symmetric buffer at a given offset (NCCL).
 */
__device__ __forceinline__ void* nccl_get_peer_ptr(
    NCCLSymmContext* nccl_ctx,
    int peer,
    size_t byte_offset) {
#ifdef __CUDA_ARCH__
  return ncclGetLsaPointer(nccl_ctx->buffer_window, byte_offset, peer);
#else
  return nullptr;
#endif
}

/**
 * Synchronize all ranks using NCCL LSA barrier.
 */
__device__ __forceinline__ void nccl_lsa_barrier(
    NCCLSymmContext* nccl_ctx,
    int barrier_id) {
#ifdef __CUDA_ARCH__
  ncclLsaBarrier(nccl_ctx->dev_comm, barrier_id);
#endif
}

/**
 * NCCL backend implementation of all-reduce sum (float32).
 *
 * Algorithm:
 * 1. Barrier to ensure all data is ready
 * 2. Each rank reads from all other ranks and accumulates
 * 3. Barrier to ensure all reads are complete
 */
__device__ int32_t nccl_all_reduce_sum_f32_impl(
    NCCLSymmContext* nccl_ctx,
    float* local_buffer,
    int64_t byte_offset,
    int64_t num_elements) {
  int rank = nccl_ctx->rank;
  int world_size = nccl_ctx->world_size;

  // Calculate thread indices
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Barrier ID for synchronization
  int barrier_id = 0;

#ifdef __CUDA_ARCH__
  // Barrier before reading - ensure all ranks have written their data
  nccl_lsa_barrier(nccl_ctx, barrier_id);

  // Read from all peers and accumulate
  for (int64_t i = tid; i < num_elements; i += stride) {
    float sum = local_buffer[i];

    for (int peer = 0; peer < world_size; peer++) {
      if (peer != rank) {
        float* peer_buffer =
            static_cast<float*>(nccl_get_peer_ptr(nccl_ctx, peer, byte_offset));
        sum += peer_buffer[i];
      }
    }

    local_buffer[i] = sum;
  }

  // Barrier after writing
  nccl_lsa_barrier(nccl_ctx, barrier_id + 1);
#endif

  return 0;
}

#endif // NCCL_HAS_DEVICE_BITCODE

// =============================================================================
// NVSHMEM BACKEND IMPLEMENTATION
// This is fully functional with libnvshmem_device.bc
// =============================================================================

/**
 * Cast SymmContext to NVSHMEMSymmContext after runtime type check.
 * Returns nullptr if the context is not NVSHMEM type.
 */
__device__ __forceinline__ NVSHMEMSymmContext* cast_to_nvshmem_context(
    SymmContext* ctx) {
  if (ctx == nullptr || ctx->type != SymmContext::Type::NVSHMEM) {
    return nullptr;
  }
  return static_cast<NVSHMEMSymmContext*>(ctx);
}

/**
 * NVSHMEM backend implementation of all-reduce sum (float32).
 *
 * Algorithm (two-phase to avoid race conditions):
 * 1. Block-level barrier to ensure all PEs have their data ready
 * 2. Each PE reads from all peers and computes local sum (read phase)
 * 3. Block-level barrier to ensure all PEs have finished reading
 * 4. Each PE writes the computed sum to its local buffer (write phase)
 * 5. Block-level barrier to ensure all writes are complete
 *
 * This implementation uses nvshmem_ptr() to get direct pointers to peer
 * symmetric memory and performs the reduction locally.
 *
 * IMPORTANT: Uses global PE numbers (from context's
 * global_rank/global_world_size), not group-local ranks. NVSHMEM's
 * nvshmem_ptr() expects global PE numbers.
 */
__device__ int32_t nvshmem_all_reduce_sum_f32_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    float* local_buffer,
    int32_t byte_offset,
    int32_t num_elements) {
  // Use global PE numbers, not group-local ranks
  int global_rank = nvshmem_ctx->global_rank;
  int global_world_size = nvshmem_ctx->global_world_size;

  // Calculate thread indices
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

#ifdef __CUDA_ARCH__
  // Barrier before reading - ensure all PEs have written their data
  nvshmemx_barrier_all_block();

  // Get the base symmetric address for this buffer
  void* symm_base = nvshmem_ctx->local_buffer;

  // Process elements in chunks to avoid using too many registers
  for (int32_t i = tid; i < num_elements; i += stride) {
    // Phase 1: Read and accumulate from all peers
    float sum = local_buffer[i];

    for (int pe = 0; pe < global_world_size; pe++) {
      if (pe != global_rank) {
        // Get pointer to peer's symmetric memory using GLOBAL PE number
        float* peer_base = static_cast<float*>(nvshmem_ptr(symm_base, pe));
        if (peer_base != nullptr) {
          // Apply the byte offset and element index
          float* peer_buffer = reinterpret_cast<float*>(
              reinterpret_cast<char*>(peer_base) + byte_offset);
          sum += peer_buffer[i];
        }
      }
    }

    // Phase 2: Barrier to ensure all PEs have finished reading before any
    // writes
    nvshmemx_barrier_all_block();

    // Phase 3: Write the result
    local_buffer[i] = sum;

    // Phase 4: Barrier before next iteration (if any)
    // This ensures writes are visible before next read phase
    if (i + stride < num_elements) {
      nvshmemx_barrier_all_block();
    }
  }

  // Final barrier to ensure all writes are complete
  nvshmemx_barrier_all_block();
#endif

  return 0;
}

// =============================================================================
// UNIFIED FRONTEND - DISPATCHES TO APPROPRIATE BACKEND
// =============================================================================

/**
 * Unified all-reduce sum implementation that dispatches to the appropriate
 * backend based on the context type.
 *
 * @param ctx Pointer to SymmContext (either NCCL or NVSHMEM)
 * @param local_buffer Pointer to local buffer containing input and output
 * @param byte_offset Byte offset within the symmetric buffer
 * @param num_elements Number of float32 elements to reduce
 * @return 0 on success, negative on error
 */
__device__ int32_t symm_all_reduce_sum_f32_impl(
    SymmContext* ctx,
    float* local_buffer,
    int32_t byte_offset,
    int32_t num_elements) {
  if (ctx == nullptr) {
    return -1; // Error: null context
  }

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_all_reduce_sum_f32_impl(
          nccl_ctx, local_buffer, byte_offset, num_elements);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_all_reduce_sum_f32_impl(
          nvshmem_ctx, local_buffer, byte_offset, num_elements);
    }
    default:
      return -2; // Error: unknown context type (or NCCL without device bitcode)
  }
}

// =============================================================================
// TRITON EXTERN_ELEMENTWISE WRAPPER
// Single entry point for Triton kernels
// =============================================================================

/**
 * Unified wrapper for all-reduce sum (float32) for use with Triton.
 *
 * This function serves as the single entry point for Triton kernels.
 * It dispatches to either NCCL or NVSHMEM backend based on the context type.
 *
 * Note: This is a collective operation that must be called by all ranks/PEs.
 * The context, offset, and num_elements should be the same across all ranks.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64 for Triton compatibility)
 * @param local_ptr Pointer to local buffer (as int64 for Triton compatibility)
 * @param byte_offset Byte offset within symmetric buffer (int32 for Triton)
 * @param num_elements Number of elements to reduce (int32 for Triton)
 * @return 0 on success, non-zero on error
 */
__device__ int32_t symm_all_reduce_sum_f32(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int32_t byte_offset,
    int32_t num_elements) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  float* buffer = reinterpret_cast<float*>(local_ptr);
  return symm_all_reduce_sum_f32_impl(
      ctx, buffer, (int64_t)byte_offset, (int64_t)num_elements);
}

// =============================================================================
// BACKEND-SPECIFIC WRAPPERS (for direct use when backend is known)
// =============================================================================

#if NCCL_HAS_DEVICE_BITCODE
/**
 * NCCL-specific wrapper for all-reduce sum (float32).
 * Use this when you know the context is NCCL type.
 */
__device__ int32_t nccl_symm_all_reduce_sum_f32(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t byte_offset,
    int64_t num_elements) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  if (nccl_ctx == nullptr) {
    return -1;
  }
  float* buffer = reinterpret_cast<float*>(local_ptr);
  return nccl_all_reduce_sum_f32_impl(
      nccl_ctx, buffer, byte_offset, num_elements);
}
#endif // NCCL_HAS_DEVICE_BITCODE

/**
 * NVSHMEM-specific wrapper for all-reduce sum (float32).
 * Use this when you know the context is NVSHMEM type.
 *
 * Note: This function is now deprecated in favor of using the unified
 * symm_all_reduce_sum_f32 function. It is kept for backward compatibility.
 */
__device__ int32_t nvshmem_symm_all_reduce_sum_f32(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int32_t byte_offset,
    int32_t num_elements) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  if (nvshmem_ctx == nullptr) {
    return -1;
  }
  float* buffer = reinterpret_cast<float*>(local_ptr);
  return nvshmem_all_reduce_sum_f32_impl(
      nvshmem_ctx, buffer, (int64_t)byte_offset, (int64_t)num_elements);
}

} // extern "C"
