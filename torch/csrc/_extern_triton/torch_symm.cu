// torch_symm.cu
// Unified CUDA device functions for symmetric memory operations
//
// This file contains unified frontend device functions that dispatch to either
// NCCL or NVSHMEM backends based on the SymmContext type.
//
// ARCHITECTURE:
// - nccl_symm.cuh: NCCL backend implementations (always compiled)
// - nvshmem_symm.cuh: NVSHMEM backend implementations (always compiled)
// - torch_symm.cu: Unified frontend that dispatches to backends
//
// The backend-specific files are always compiled to catch any compilation
// errors. The NCCL_HAS_DEVICE_BITCODE flag controls whether NCCL dispatch
// is enabled in the unified frontend (since NCCL doesn't provide a device
// bitcode library for Triton linking).
//
// BACKEND SUPPORT:
// - NVSHMEM: Fully functional. NVSHMEM provides libnvshmem_device.bc.
// - NCCL: Compiles but not functional without libnccl_device.bc.
//
// Compile to bitcode:
//   clang++ -x cuda --cuda-device-only -emit-llvm -c torch_symm.cu \
//           -o torch_symm.bc --cuda-gpu-arch=sm_80 -O3 \
//           -fcuda-flush-denormals-to-zero

#include <cuda_runtime.h>
#include <stdint.h>

// Include the unified context definitions
#include "symm_comm.cuh"

// Include backend implementations (always compiled for error checking)
#include "nccl_symm.cuh"
#include "nvshmem_symm.cuh"

// =============================================================================
// NCCL_HAS_DEVICE_BITCODE CONFIGURATION
// =============================================================================
// Set to 0 because NCCL does not ship a device bitcode library.
// When set to 1, the unified frontend will dispatch to NCCL backend.
// The NCCL backend code itself is always compiled (in nccl_symm.cuh).
#ifndef NCCL_HAS_DEVICE_BITCODE
#define NCCL_HAS_DEVICE_BITCODE 0
#endif

// Mark functions as extern "C" to avoid C++ name mangling
extern "C" {

// =============================================================================
// TRITON EXTERN_ELEMENTWISE ENTRY POINTS
// Unified frontend that dispatches to appropriate backend
// =============================================================================

/**
 * Unified all-reduce that dispatches to the appropriate backend.
 *
 * DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
 * NOT efficient.
 */
__device__ int32_t symm_all_reduce(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int32_t byte_offset,
    int32_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  float* buffer = reinterpret_cast<float*>(local_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_all_reduce_impl(
          nccl_ctx, buffer, byte_offset, num_elements, reduce_op, dtype);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_all_reduce_impl(
          nvshmem_ctx, buffer, byte_offset, num_elements, reduce_op, dtype);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified quiet operation.
 */
__device__ int32_t symm_quiet(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_quiet_impl(nccl_ctx);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_quiet_impl(nvshmem_ctx);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified barrier operation.
 */
__device__ int32_t symm_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_barrier_impl(nccl_ctx, 0);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_barrier_impl(nvshmem_ctx);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified fence operation.
 */
__device__ int32_t symm_fence(int64_t ctx_ptr, int32_t scope) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_fence_impl(nccl_ctx, scope);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_fence_impl(nvshmem_ctx, scope);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified lsa_barrier operation.
 *
 * Performs barrier synchronization among ranks in the same LSA (Local Symmetric
 * Access) domain. Only ranks that can directly access each other's memory via
 * load/store operations participate in this barrier.
 *
 * This is useful for synchronizing within a local group (e.g., GPUs on the same
 * node with NVLink connectivity) without waiting for remote ranks.
 *
 * For NVSHMEM, uses nvshmemx_barrier_block with NVSHMEM_TEAM_SHARED.
 * For NCCL, uses ncclLsaBarrierSession with ncclTeamTagLsa.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param barrier_index Index of the barrier to use (0..nBarriers-1).
 *        Multiple independent barriers can be used by specifying different
 * indices.
 * @return 0 on success
 */
__device__ int32_t symm_lsa_barrier(int64_t ctx_ptr, int32_t barrier_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_lsa_barrier_impl(nccl_ctx, barrier_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_lsa_barrier_impl(nvshmem_ctx, barrier_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified lsa_barrier_arrive operation.
 *
 * Signals arrival at an LSA barrier without waiting for other peers.
 * This is the "arrive" phase of a split-phase barrier, allowing overlapping
 * computation while waiting for peers.
 *
 * This is useful for split-phase synchronization where:
 * 1. Ranks signal arrival (symm_lsa_barrier_arrive)
 * 2. Ranks do other work
 * 3. Ranks wait for all peers (symm_lsa_barrier_wait)
 *
 * For NCCL, uses ncclLsaBarrierSession with its arrive() method.
 * For NVSHMEM, uses signal operations on the LSA signal pad to notify
 * all peers of arrival.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param barrier_index Index of the barrier to use (0..nBarriers-1).
 *        Multiple independent barriers can be used by specifying different
 * indices. The same index must be used for the corresponding wait() call.
 * @return 0 on success
 */
__device__ int32_t
symm_lsa_barrier_arrive(int64_t ctx_ptr, int32_t barrier_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_lsa_barrier_arrive_impl(nccl_ctx, barrier_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_lsa_barrier_arrive_impl(nvshmem_ctx, barrier_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified lsa_barrier_wait operation.
 *
 * Waits for all peers to arrive at the LSA barrier.
 * This is the "wait" phase of a split-phase barrier.
 *
 * Must be called after symm_lsa_barrier_arrive to complete the barrier.
 * After this returns, all data written by peers before their arrive() call
 * is guaranteed to be visible.
 *
 * For NCCL, uses ncclLsaBarrierSession with its wait() method.
 * For NVSHMEM, waits until the local barrier signal reaches the expected
 * count, then resets the signal for the next iteration.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param barrier_index Index of the barrier to wait on (0..nBarriers-1).
 *        Must match the index used in the corresponding arrive() call.
 * @return 0 on success
 */
__device__ int32_t
symm_lsa_barrier_wait(int64_t ctx_ptr, int32_t barrier_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_lsa_barrier_wait_impl(nccl_ctx, barrier_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_lsa_barrier_wait_impl(nvshmem_ctx, barrier_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

// =============================================================================
// GIN (GPU-Initiated Networking) BARRIER PRIMITIVES
// These barrier primitives target the full team (all ranks), not just LSA
// domain
// =============================================================================

/**
 * Unified barrier_arrive operation (GIN-based, full team).
 *
 * Signals arrival at a GIN barrier without waiting for other peers.
 * This is the "arrive" phase of a split-phase barrier, allowing overlapping
 * computation while waiting for peers.
 *
 * Unlike symm_lsa_barrier_arrive which only targets the LSA domain, this GIN
 * barrier targets ALL ranks in the team using GPU-initiated networking.
 *
 * This is useful for split-phase synchronization across all ranks where:
 * 1. Ranks signal arrival (symm_barrier_arrive)
 * 2. Ranks do other work
 * 3. Ranks wait for all peers (symm_barrier_wait)
 *
 * For NCCL, uses ncclLsaBarrierSession with ncclTeamTagWorld.
 * For NVSHMEM, uses signal operations on the GIN signal pad to notify
 * all peers of arrival.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param barrier_index Index of the barrier to use (0..nBarriers-1).
 *        Multiple independent barriers can be used by specifying different
 * indices. The same index must be used for the corresponding wait() call.
 * @return 0 on success
 */
__device__ int32_t symm_barrier_arrive(int64_t ctx_ptr, int32_t barrier_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_barrier_arrive_impl(nccl_ctx, barrier_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_barrier_arrive_impl(nvshmem_ctx, barrier_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified barrier_wait operation (GIN-based, full team).
 *
 * Waits for all peers to arrive at the GIN barrier.
 * This is the "wait" phase of a split-phase barrier.
 *
 * Unlike symm_lsa_barrier_wait which only waits for LSA domain peers, this GIN
 * barrier waits for ALL ranks in the team.
 *
 * Must be called after symm_barrier_arrive to complete the barrier.
 * After this returns, all data written by peers before their arrive() call
 * is guaranteed to be visible.
 *
 * For NCCL, uses ncclLsaBarrierSession with ncclTeamTagWorld.
 * For NVSHMEM, waits until the local GIN barrier signal reaches the expected
 * count, then updates the epoch for the next iteration.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param barrier_index Index of the barrier to wait on (0..nBarriers-1).
 *        Must match the index used in the corresponding arrive() call.
 * @return 0 on success
 */
__device__ int32_t symm_barrier_wait(int64_t ctx_ptr, int32_t barrier_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_barrier_wait_impl(nccl_ctx, barrier_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_barrier_wait_impl(nvshmem_ctx, barrier_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified lsa_ptr operation.
 */
__device__ int64_t
symm_remote_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_lsa_ptr_impl(nccl_ctx, local_ptr, peer);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_lsa_ptr_impl(nvshmem_ctx, local_ptr, peer);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified lsa_multicast_ptr operation.
 *
 * Gets the team from context and uses it for multicast operations.
 */
__device__ int64_t symm_multicast_ptr(int64_t ctx_ptr, int64_t local_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_lsa_multicast_ptr_impl(nccl_ctx, local_ptr);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      TORCH_SYMM_CHECK(nvshmem_ctx->team != nullptr, "team is null in context");
      return nvshmem_lsa_multicast_ptr_impl(
          nvshmem_ctx, local_ptr, nvshmem_ctx->team);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified signal operation.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * This is a point-to-point notification mechanism without data transfer.
 * Uses the signal pad stored in the context.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination rank to signal
 * @param value Value to use in the operation
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 * @return 0 on success
 */
__device__ int32_t symm_signal(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t dest_rank,
    int64_t value,
    int32_t op) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_signal_impl(
          nccl_ctx, signal_index, dest_rank, static_cast<uint64_t>(value), op);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_signal_impl(
          nvshmem_ctx,
          signal_index,
          dest_rank,
          static_cast<uint64_t>(value),
          op);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified lsa_signal_ptr operation.
 *
 * Returns a device pointer to a peer's signal pad, if accessible via P2P/LSA.
 * This allows direct load/store access to the peer's signal memory.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param peer Peer rank to get signal pad pointer for
 * @return Device pointer to peer's signal pad, or 0 if not accessible
 */
__device__ int64_t symm_signal_ptr(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_lsa_signal_ptr_impl(nccl_ctx, peer);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_lsa_signal_ptr_impl(nvshmem_ctx, peer);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified signal_wait_until operation.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Supported conditions:
 * - SIGNAL_CMP_EQ (1): Wait until signal == cmp_value
 * - SIGNAL_CMP_NE (2): Wait until signal != cmp_value
 * - SIGNAL_CMP_GT (3): Wait until signal > cmp_value
 * - SIGNAL_CMP_GE (4): Wait until signal >= cmp_value
 * - SIGNAL_CMP_LT (5): Wait until signal < cmp_value
 * - SIGNAL_CMP_LE (6): Wait until signal <= cmp_value
 *
 * Note: NCCL only supports SIGNAL_CMP_GE condition.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ int64_t symm_signal_wait_until(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t cmp,
    int64_t cmp_value) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return static_cast<int64_t>(nccl_signal_wait_until_impl(
          nccl_ctx, signal_index, cmp, static_cast<uint64_t>(cmp_value)));
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return static_cast<int64_t>(nvshmem_signal_wait_until_impl(
          nvshmem_ctx, signal_index, cmp, static_cast<uint64_t>(cmp_value)));
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified signal_reset operation.
 *
 * Resets a local signal at signal_index to zero. This is used to prepare
 * a signal for the next round of signaling/waiting in iterative algorithms.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to reset
 * @return 0 on success
 */
__device__ int32_t symm_signal_reset(int64_t ctx_ptr, int32_t signal_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_signal_reset_impl(nccl_ctx, signal_index);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_signal_reset_impl(nvshmem_ctx, signal_index);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified put_async operation.
 *
 * Non-blocking one-sided put: copies count elements of element_size bytes
 * from src_ptr (local) to dest_ptr (symmetric address) on dest_rank's buffer.
 *
 * Returns immediately without waiting for completion. Use symm_quiet to
 * ensure all prior put operations have completed.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param dest_ptr Destination pointer (symmetric address, as int64)
 * @param src_ptr Source pointer (local address, as int64)
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination rank/PE number
 * @return 0 on success
 */
__device__ int32_t symm_put_async(
    int64_t ctx_ptr,
    int64_t dest_ptr,
    int64_t src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_put_async_impl(nccl_ctx, dest, src, count, element_size, dest_rank);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_put_async_impl(
          nvshmem_ctx, dest, src, count, element_size, dest_rank);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

/**
 * Unified put_signal_async operation.
 *
 * Non-blocking one-sided put with remote signal: copies count elements of
 * element_size bytes from src_ptr (local) to dest_ptr (symmetric address) on
 * dest_rank's buffer. After the data transfer completes, atomically updates
 * the remote signal at signal_index on dest_rank using the specified operation.
 *
 * This is a fused operation that combines data transfer with signaling,
 * allowing the receiver to know when the data has arrived without polling.
 * The signal update is guaranteed to be visible only after the data transfer
 * is complete.
 *
 * Returns immediately without waiting for completion. Use symm_quiet to
 * ensure all prior put operations have completed.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param dest_ptr Destination pointer (symmetric address, as int64)
 * @param src_ptr Source pointer (local address, as int64)
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination rank/PE number
 * @param signal_index Index into the signal pad to update on dest_rank
 * @param signal_value Value to use in the signal operation
 * @param signal_op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 * @return 0 on success
 */
__device__ int32_t symm_put_signal_async(
    int64_t ctx_ptr,
    int64_t dest_ptr,
    int64_t src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank,
    int32_t signal_index,
    int64_t signal_value,
    int32_t signal_op) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_put_signal_async_impl(
          nccl_ctx,
          dest,
          src,
          count,
          element_size,
          dest_rank,
          signal_index,
          static_cast<uint64_t>(signal_value),
          signal_op);
      return 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_put_signal_async_impl(
          nvshmem_ctx,
          dest,
          src,
          count,
          element_size,
          dest_rank,
          signal_index,
          static_cast<uint64_t>(signal_value),
          signal_op);
      return 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return -1;
  }
}

// =============================================================================
// SYMM_TEAM PRIMITIVES - TOPOLOGY MANAGEMENT
// =============================================================================

/**
 * Get the number of ranks in the team.
 * Team is obtained from the context.
 */
__device__ int32_t symm_team_size(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nccl_ctx->team != nullptr, "team is null in NCCL context");
      return nccl_ctx->team->team_size;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nvshmem_ctx->team != nullptr, "team is null in NVSHMEM context");
      return nvshmem_ctx->team->team_size;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Get the calling process's rank index within the team.
 * Team is obtained from the context.
 */
__device__ int32_t symm_team_rank(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nccl_ctx->team != nullptr, "team is null in NCCL context");
      return nccl_ctx->team->team_rank;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nvshmem_ctx->team != nullptr, "team is null in NVSHMEM context");
      return nvshmem_ctx->team->team_rank;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Get the number of ranks in the caller's LSA domain.
 * Team is obtained from the context.
 */
__device__ int32_t symm_team_lsa_size(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nccl_ctx->team != nullptr, "team is null in NCCL context");
      return nccl_ctx->team->lsa_size;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nvshmem_ctx->team != nullptr, "team is null in NVSHMEM context");
      return nvshmem_ctx->team->lsa_size;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Check if a peer rank is in the same LSA domain as the caller.
 * Team is obtained from the context.
 */
__device__ int32_t symm_team_lsa(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nccl_ctx->team != nullptr, "team is null in NCCL context");
      return nccl_ctx->team->is_lsa_peer(peer) ? 1 : 0;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      TORCH_SYMM_CHECK(
          nvshmem_ctx->team != nullptr, "team is null in NVSHMEM context");
      return nvshmem_ctx->team->is_lsa_peer(peer) ? 1 : 0;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

} // extern "C"
