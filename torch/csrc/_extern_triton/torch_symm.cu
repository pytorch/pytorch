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
// UNIFIED FRONTEND - DISPATCHES TO APPROPRIATE BACKEND
// =============================================================================

/**
 * Unified all-reduce implementation that dispatches to the appropriate
 * backend based on the context type.
 *
 * DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
 * NOT efficient.
 */
__device__ void symm_all_reduce_impl(
    SymmContext* ctx,
    float* local_buffer,
    int32_t byte_offset,
    int32_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_all_reduce_impl(
          nccl_ctx, local_buffer, byte_offset, num_elements, reduce_op, dtype);
      return;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_all_reduce_impl(
          nvshmem_ctx,
          local_buffer,
          byte_offset,
          num_elements,
          reduce_op,
          dtype);
      return;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
  }
}

/**
 * Unified quiet implementation.
 */
__device__ void symm_quiet_impl(SymmContext* ctx) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_quiet_impl(nccl_ctx);
      return;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_quiet_impl(nvshmem_ctx);
      return;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
  }
}

/**
 * Unified barrier implementation.
 */
__device__ void symm_barrier_impl(SymmContext* ctx) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_barrier_impl(nccl_ctx, 0);
      return;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_barrier_impl(nvshmem_ctx);
      return;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
  }
}

/**
 * Unified fence implementation.
 */
__device__ void symm_fence_impl(SymmContext* ctx, int32_t scope) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_fence_impl(nccl_ctx, scope);
      return;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_fence_impl(nvshmem_ctx, scope);
      return;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
  }
}

/**
 * Unified remote_ptr implementation.
 */
__device__ int64_t
symm_remote_ptr_impl(SymmContext* ctx, int64_t local_ptr, int32_t peer) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_remote_ptr_impl(nccl_ctx, local_ptr, peer);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_remote_ptr_impl(nvshmem_ctx, local_ptr, peer);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified multicast_ptr implementation.
 */
__device__ int64_t
symm_multicast_ptr_impl(SymmContext* ctx, int64_t local_ptr, SymmTeam* team) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_multicast_ptr_impl(nccl_ctx, local_ptr);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
      return nvshmem_multicast_ptr_impl(nvshmem_ctx, local_ptr, nvshmem_team);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified signal implementation.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * This is a point-to-point notification mechanism without data transfer.
 * Gets the signal pad from the context and computes the offset for the signal.
 *
 * @param ctx SymmContext pointer
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination rank to signal
 * @param value Value to use in the operation
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ void symm_signal_impl(
    SymmContext* ctx,
    int32_t signal_index,
    int32_t dest_rank,
    uint64_t value,
    int32_t op) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      nccl_signal_impl(nccl_ctx, signal_index, dest_rank, value, op);
      return;
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      nvshmem_signal_impl(nvshmem_ctx, signal_index, dest_rank, value, op);
      return;
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
  }
}

/**
 * Unified signal_ptr implementation.
 *
 * Returns a device pointer to a peer's signal pad, if accessible via P2P.
 * This allows direct load/store access to the peer's signal memory.
 *
 * For NVSHMEM, this returns the LSA signal pad (for P2P load/store access).
 * For NCCL, this uses the LSA window to get the peer's signal pad address.
 *
 * @param ctx SymmContext pointer
 * @param peer Peer rank to get signal pad pointer for
 * @return Device pointer to peer's signal pad, or 0 if not accessible
 */
__device__ int64_t symm_signal_ptr_impl(SymmContext* ctx, int32_t peer) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_signal_ptr_impl(nccl_ctx, peer);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_signal_ptr_impl(nvshmem_ctx, peer);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

/**
 * Unified signal_wait_until implementation.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Uses the gin_signal_pad from the context (same pad that symm_signal writes
 * to). This enables point-to-point synchronization patterns where one rank
 * signals with symm_signal and another waits with symm_signal_wait_until.
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
 * @param ctx SymmContext pointer
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ uint64_t symm_signal_wait_until_impl(
    SymmContext* ctx,
    int32_t signal_index,
    int32_t cmp,
    uint64_t cmp_value) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");

  switch (ctx->type) {
#if NCCL_HAS_DEVICE_BITCODE
    case SymmContext::Type::NCCL: {
      NCCLSymmContext* nccl_ctx = static_cast<NCCLSymmContext*>(ctx);
      return nccl_signal_wait_until_impl(
          nccl_ctx, signal_index, cmp, cmp_value);
    }
#endif
    case SymmContext::Type::NVSHMEM: {
      NVSHMEMSymmContext* nvshmem_ctx = static_cast<NVSHMEMSymmContext*>(ctx);
      return nvshmem_signal_wait_until_impl(
          nvshmem_ctx, signal_index, cmp, cmp_value);
    }
    default:
      TORCH_SYMM_CHECK(false, "Unknown SymmContext type");
      return 0;
  }
}

// =============================================================================
// TRITON EXTERN_ELEMENTWISE WRAPPERS
// Single entry points for Triton kernels
// =============================================================================

/**
 * Unified wrapper for all-reduce for use with Triton.
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
  symm_all_reduce_impl(
      ctx, buffer, byte_offset, num_elements, reduce_op, dtype);
  return 0;
}

/**
 * Unified wrapper for quiet operation for use with Triton.
 */
__device__ int32_t symm_quiet(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  symm_quiet_impl(ctx);
  return 0;
}

/**
 * Unified wrapper for barrier operation for use with Triton.
 */
__device__ int32_t symm_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  symm_barrier_impl(ctx);
  return 0;
}

/**
 * Unified wrapper for fence operation for use with Triton.
 */
__device__ int32_t symm_fence(int64_t ctx_ptr, int32_t scope) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  symm_fence_impl(ctx, scope);
  return 0;
}

/**
 * Unified wrapper for remote_ptr operation for use with Triton.
 */
__device__ int64_t
symm_remote_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  return symm_remote_ptr_impl(ctx, local_ptr, peer);
}

/**
 * Unified wrapper for multicast_ptr operation for use with Triton.
 */
__device__ int64_t
symm_multicast_ptr(int64_t ctx_ptr, int64_t local_ptr, int64_t team_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  return symm_multicast_ptr_impl(ctx, local_ptr, team);
}

/**
 * Unified wrapper for signal operation for use with Triton.
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
 */
__device__ int32_t symm_signal(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t dest_rank,
    int64_t value,
    int32_t op) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  symm_signal_impl(
      ctx, signal_index, dest_rank, static_cast<uint64_t>(value), op);
  return 0;
}

/**
 * Unified wrapper for signal_ptr operation for use with Triton.
 *
 * Returns a device pointer to a peer's signal pad, if accessible via P2P.
 * This allows direct load/store access to the peer's signal memory.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param peer Peer rank to get signal pad pointer for
 * @return Device pointer to peer's signal pad, or 0 if not accessible
 */
__device__ int64_t symm_signal_ptr(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  return symm_signal_ptr_impl(ctx, peer);
}

/**
 * Unified wrapper for signal_wait_until operation for use with Triton.
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
  return static_cast<int64_t>(symm_signal_wait_until_impl(
      ctx, signal_index, cmp, static_cast<uint64_t>(cmp_value)));
}

// =============================================================================
// SYMM_TEAM PRIMITIVES - TOPOLOGY MANAGEMENT
// =============================================================================

/**
 * Get the number of ranks in the team.
 */
__device__ int32_t symm_team_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  return team->team_size;
}

/**
 * Get the calling process's rank index within the team.
 */
__device__ int32_t symm_team_rank(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  return team->team_rank;
}

/**
 * Get the number of ranks in the caller's LSA domain.
 */
__device__ int32_t symm_team_lsa_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  return team->lsa_size;
}

/**
 * Check if a peer rank is in the same LSA domain as the caller.
 */
__device__ int32_t symm_team_lsa(int64_t team_ptr, int32_t peer) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  return team->is_lsa_peer(peer) ? 1 : 0;
}

} // extern "C"
