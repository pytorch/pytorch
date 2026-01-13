// symm_all_reduce.cu
// CUDA device functions for symmetric memory all-reduce operations
//
// This file contains device functions that can be compiled to LLVM bitcode
// (.bc) and linked with Triton kernels via extern_libs.
//
// The implementation follows the NCCL Simple LSA Kernel pattern, abstracting
// NCCL-specific types behind a generic SymmContext interface.
//
// IMPORTANT LIMITATION:
// This implementation requires NCCL device functions (ncclLsaBarrier,
// ncclGetLsaPointer) which are declared as extern. Unlike NVSHMEM which
// provides a libnvshmem_device.bc file, NCCL does not ship a device bitcode
// library. As a result, this file will compile to bitcode but the symbols
// will remain unresolved at PTX generation time.
//
// For this to work, one of the following would be needed:
// 1. NCCL provides a libnccl_device.bc file (similar to NVSHMEM)
// 2. The implementation is changed to avoid NCCL device functions
// 3. A different linking mechanism is used (e.g., JIT compilation at runtime)
//
// Compile to bitcode:
//   clang++ -x cuda --cuda-device-only -emit-llvm -c symm_all_reduce.cu \
//           -o symm_all_reduce.bc --cuda-gpu-arch=sm_80 -O3 \
//           -fcuda-flush-denormals-to-zero

#include <cuda_runtime.h>
#include <stdint.h>

// Include the context definitions
#include <nccl_symm_comm.cuh>

// NCCL headers are needed for the implementation
// These are available when building with NCCL support
#ifdef __CUDA_ARCH__
// Device-side NCCL functions - forward declarations
// The actual symbols come from NCCL library linked at runtime
extern "C" __device__ void* ncclGetLsaPointer(
    void* window,
    size_t offset,
    int peer);
extern "C" __device__ void ncclLsaBarrier(void* devComm, int barrierId);
#endif

// Mark functions as extern "C" to avoid C++ name mangling
extern "C" {

// =============================================================================
// HELPER DEVICE FUNCTIONS
// =============================================================================

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
 * Get a pointer to a peer's symmetric buffer at a given offset.
 *
 * @param nccl_ctx NCCL symmetric context
 * @param peer Peer rank to get pointer for
 * @param byte_offset Byte offset within the buffer
 * @return Pointer to the peer's buffer at the given offset
 */
__device__ __forceinline__ void* get_peer_ptr(
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
 *
 * @param nccl_ctx NCCL symmetric context
 * @param barrier_id Barrier identifier (should be unique per synchronization
 * point)
 */
__device__ __forceinline__ void lsa_barrier(
    NCCLSymmContext* nccl_ctx,
    int barrier_id) {
#ifdef __CUDA_ARCH__
  ncclLsaBarrier(nccl_ctx->dev_comm, barrier_id);
#endif
}

// =============================================================================
// SYMMETRIC ALL-REDUCE KERNEL
// Implements a simple ring all-reduce using LSA (Local Symmetric Access)
// =============================================================================

/**
 * Perform all-reduce sum operation on symmetric memory buffers (float32).
 *
 * This kernel implements a simple ring all-reduce algorithm using NCCL's
 * Local Symmetric Access (LSA) API. Each rank reads from its neighbors
 * and accumulates the values locally.
 *
 * Algorithm:
 * 1. Barrier to ensure all data is ready
 * 2. Each rank reads from all other ranks and accumulates
 * 3. Barrier to ensure all reads are complete
 *
 * @param ctx Pointer to SymmContext (will be cast to NCCLSymmContext)
 * @param local_buffer Pointer to local buffer containing input and output
 * @param byte_offset Byte offset within the symmetric buffer where data starts
 * @param num_elements Number of float32 elements to reduce
 * @return 0 on success, non-zero on error
 */
__device__ int32_t symm_all_reduce_sum_f32_impl(
    SymmContext* ctx,
    float* local_buffer,
    int64_t byte_offset,
    int64_t num_elements) {
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  if (nccl_ctx == nullptr) {
    return -1; // Error: invalid context
  }

  int rank = nccl_ctx->rank;
  int world_size = nccl_ctx->world_size;

  // Calculate thread indices
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Barrier ID for synchronization (use 0 for simplicity)
  // In production, this should be passed as a parameter
  int barrier_id = 0;

#ifdef __CUDA_ARCH__
  // Barrier before reading - ensure all ranks have written their data
  lsa_barrier(nccl_ctx, barrier_id);

  // Read from all peers and accumulate
  // Each thread handles multiple elements in a grid-stride loop
  for (int64_t i = tid; i < num_elements; i += stride) {
    float sum = local_buffer[i]; // Start with local value

    // Accumulate values from all other ranks
    for (int peer = 0; peer < world_size; peer++) {
      if (peer != rank) {
        float* peer_buffer =
            static_cast<float*>(get_peer_ptr(nccl_ctx, peer, byte_offset));
        sum += peer_buffer[i];
      }
    }

    // Write result back to local buffer
    local_buffer[i] = sum;
  }

  // Barrier after writing - ensure all ranks have finished
  lsa_barrier(nccl_ctx, barrier_id + 1);
#endif

  return 0; // Success
}

// =============================================================================
// SCALAR WRAPPER FOR TRITON EXTERN_ELEMENTWISE
// This function matches the signature expected by Triton's extern_elementwise
// =============================================================================

/**
 * Scalar wrapper for all-reduce sum (float32).
 *
 * Note: This is a collective operation that must be called by all ranks.
 * The context, offset, and num_elements should be the same across all ranks.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64 for Triton compatibility)
 * @param local_ptr Pointer to local buffer (as int64 for Triton compatibility)
 * @param byte_offset Byte offset within symmetric buffer
 * @param num_elements Number of elements to reduce
 * @return 0 on success, non-zero on error
 */
__device__ int32_t symm_all_reduce_sum_f32(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t byte_offset,
    int64_t num_elements) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  float* buffer = reinterpret_cast<float*>(local_ptr);
  return symm_all_reduce_sum_f32_impl(ctx, buffer, byte_offset, num_elements);
}

} // extern "C"
