// nvshmem_symm.cuh
// NVSHMEM backend implementation for symmetric memory operations
//
// This file contains NVSHMEM-specific device functions that can be compiled
// independently and linked into the unified torch_symm.bc bitcode library.
//
// NVSHMEM provides libnvshmem_device.bc which can be linked with this
// implementation at compile time.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "symm_comm.cuh"

// =============================================================================
// REDUCTION OPERATION AND DATA TYPE CONSTANTS
// =============================================================================

#ifndef REDUCE_OP_SUM
#define REDUCE_OP_SUM 0
#endif

#ifndef DTYPE_FLOAT32
#define DTYPE_FLOAT32 0
#endif

// Fence scope constants
#ifndef FENCE_SCOPE_CTA
#define FENCE_SCOPE_CTA 0
#define FENCE_SCOPE_GPU 1
#define FENCE_SCOPE_SYSTEM 2
#endif

// Signal operation constants
// Note: These must match the Python SIGNAL_OP_* constants
// We map them to NVSHMEM's native signal operation constants internally
#ifndef SIGNAL_OP_SET
#define SIGNAL_OP_SET 0
#define SIGNAL_OP_ADD 1
#endif

// NVSHMEM native signal operation constants
// Note: These values come from nvshmem_common_transport.h:
//   NVSHMEM_SIGNAL_SET = 9, NVSHMEM_SIGNAL_ADD = 10
// We define them only if not already defined by NVSHMEM headers
#ifndef NVSHMEM_SIGNAL_SET
#define NVSHMEM_SIGNAL_SET 9
#define NVSHMEM_SIGNAL_ADD 10
#endif

// Signal comparison condition constants (for symm_signal_wait_until)
// These match our abstracted constants in symm_comm.cuh
#ifndef SIGNAL_CMP_EQ
#define SIGNAL_CMP_EQ 1 // Equal
#define SIGNAL_CMP_NE 2 // Not equal
#define SIGNAL_CMP_GT 3 // Greater than
#define SIGNAL_CMP_GE 4 // Greater than or equal
#define SIGNAL_CMP_LT 5 // Less than
#define SIGNAL_CMP_LE 6 // Less than or equal
#endif

// =============================================================================
// NVSHMEM CAST HELPERS
// =============================================================================

/**
 * Cast SymmContext to NVSHMEMSymmContext after runtime type check.
 * Asserts on failure.
 */
__device__ __forceinline__ NVSHMEMSymmContext* cast_to_nvshmem_context(
    SymmContext* ctx) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");
  TORCH_SYMM_CHECK(
      ctx->type == SymmContext::Type::NVSHMEM,
      "SymmContext is not NVSHMEM type");
  return static_cast<NVSHMEMSymmContext*>(ctx);
}

/**
 * Cast SymmTeam to NVSHMEMSymmTeam after runtime type check.
 * Asserts on failure.
 */
__device__ __forceinline__ NVSHMEMSymmTeam* cast_to_nvshmem_team(
    SymmTeam* team) {
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  TORCH_SYMM_CHECK(
      team->type == SymmTeam::Type::NVSHMEM, "SymmTeam is not NVSHMEM type");
  return static_cast<NVSHMEMSymmTeam*>(team);
}

// =============================================================================
// NVSHMEM BACKEND IMPLEMENTATIONS
// =============================================================================

/**
 * NVSHMEM backend implementation of all-reduce.
 *
 * DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
 * NOT efficient.
 */
__device__ void nvshmem_all_reduce_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    float* local_buffer,
    int32_t byte_offset,
    int32_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  TORCH_SYMM_CHECK(
      reduce_op == REDUCE_OP_SUM, "Only REDUCE_OP_SUM is supported");
  TORCH_SYMM_CHECK(dtype == DTYPE_FLOAT32, "Only DTYPE_FLOAT32 is supported");

  int global_rank = nvshmem_ctx->global_rank;
  int global_world_size = nvshmem_ctx->global_world_size;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  nvshmemx_barrier_all_block();

  void* symm_base = nvshmem_ctx->local_buffer;

  for (int32_t i = tid; i < num_elements; i += stride) {
    float sum = local_buffer[i];

    for (int pe = 0; pe < global_world_size; pe++) {
      if (pe != global_rank) {
        float* peer_base = static_cast<float*>(nvshmem_ptr(symm_base, pe));
        if (peer_base != nullptr) {
          float* peer_buffer = reinterpret_cast<float*>(
              reinterpret_cast<char*>(peer_base) + byte_offset);
          sum += peer_buffer[i];
        }
      }
    }

    nvshmemx_barrier_all_block();
    local_buffer[i] = sum;

    if (i + stride < num_elements) {
      nvshmemx_barrier_all_block();
    }
  }

  nvshmemx_barrier_all_block();
}

/**
 * NVSHMEM backend implementation of quiet.
 */
__device__ void nvshmem_quiet_impl(NVSHMEMSymmContext* nvshmem_ctx) {
  nvshmem_quiet();
}

/**
 * NVSHMEM backend implementation of barrier.
 */
__device__ void nvshmem_barrier_impl(NVSHMEMSymmContext* nvshmem_ctx) {
  nvshmemx_barrier_all_block();
}

/**
 * NVSHMEM backend implementation of fence.
 */
__device__ void nvshmem_fence_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int32_t scope) {
  switch (scope) {
    case FENCE_SCOPE_CTA:
      __syncthreads();
      nvshmem_fence();
      break;
    case FENCE_SCOPE_GPU:
      __threadfence();
      nvshmem_fence();
      break;
    case FENCE_SCOPE_SYSTEM:
      __threadfence_system();
      nvshmem_fence();
      break;
    default:
      TORCH_SYMM_CHECK(false, "Invalid fence scope");
  }
}

/**
 * NVSHMEM backend implementation of remote_ptr.
 */
__device__ int64_t nvshmem_remote_ptr_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int64_t local_ptr,
    int32_t peer) {
  void* ptr = reinterpret_cast<void*>(local_ptr);
  void* peer_ptr = nvshmem_ptr(ptr, peer);
  return reinterpret_cast<int64_t>(peer_ptr);
}

/**
 * NVSHMEM backend implementation of multicast_ptr.
 */
__device__ int64_t nvshmem_multicast_ptr_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int64_t local_ptr,
    NVSHMEMSymmTeam* nvshmem_team) {
  void* ptr = reinterpret_cast<void*>(local_ptr);
  nvshmem_team_t team = nvshmem_team->get_team_handle();
  void* mc_ptr = nvshmemx_mc_ptr(team, ptr);
  return reinterpret_cast<int64_t>(mc_ptr);
}

/**
 * NVSHMEM backend implementation of signal.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * This is a point-to-point notification mechanism without data transfer.
 * Gets the GIN signal pad from the context and computes the offset for the
 * signal. Uses the native nvshmemx_signal_op API.
 *
 * Note: This uses the gin_signal_pad (for GPU-initiated networking),
 * which is separate from lsa_signal_pad (for P2P load/store access).
 *
 * @param nvshmem_ctx NVSHMEM context with gin_signal_pad
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination PE to signal
 * @param value Value to use in the operation (default 1)
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ void nvshmem_signal_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int32_t signal_index,
    int32_t dest_rank,
    uint64_t value,
    int32_t op) {
  // Get the GIN signal pad from the context (used for remote atomic signaling)
  TORCH_SYMM_CHECK(
      nvshmem_ctx->gin_signal_pad != nullptr, "gin_signal_pad is null");
  uint64_t* signal_pad = nvshmem_ctx->gin_signal_pad;

  // Calculate the address of the specific signal
  uint64_t* target_signal = signal_pad + signal_index;

  // Map our signal operation constants to NVSHMEM's native constants
  int nvshmem_sig_op;
  switch (op) {
    case SIGNAL_OP_SET:
      nvshmem_sig_op = NVSHMEM_SIGNAL_SET;
      break;
    case SIGNAL_OP_ADD:
      nvshmem_sig_op = NVSHMEM_SIGNAL_ADD;
      break;
    default:
      TORCH_SYMM_CHECK(false, "Invalid signal operation");
      return;
  }

  // Use the native NVSHMEM signal operation API
  nvshmemx_signal_op(target_signal, value, nvshmem_sig_op, dest_rank);
}

/**
 * NVSHMEM backend implementation of signal_ptr.
 *
 * Returns a device pointer to a peer's LSA signal pad, if accessible via P2P.
 * This allows direct load/store access to the peer's signal memory.
 *
 * Note: This returns a pointer to the lsa_signal_pad (for P2P load/store
 * access), which is separate from gin_signal_pad (used by symm_signal for
 * atomic operations).
 *
 * @param nvshmem_ctx NVSHMEM context with lsa_signal_pad
 * @param peer Peer PE number to get signal pad pointer for
 * @return Device pointer to peer's LSA signal pad, or 0 if not accessible
 */
__device__ int64_t
nvshmem_signal_ptr_impl(NVSHMEMSymmContext* nvshmem_ctx, int32_t peer) {
  // Get the LSA signal pad from the context (used for P2P load/store)
  TORCH_SYMM_CHECK(
      nvshmem_ctx->lsa_signal_pad != nullptr, "lsa_signal_pad is null");

  // Use nvshmem_ptr to get the remote address (returns nullptr if not P2P
  // accessible)
  void* peer_signal_pad = nvshmem_ptr(nvshmem_ctx->lsa_signal_pad, peer);

  return reinterpret_cast<int64_t>(peer_signal_pad);
}

/**
 * NVSHMEM backend implementation of signal_wait_until.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Uses the gin_signal_pad from the context (same pad that symm_signal writes
 * to). This enables point-to-point synchronization patterns where one rank
 * signals with symm_signal and another waits with symm_signal_wait_until.
 *
 * Supported conditions (mapping SIGNAL_CMP_* to NVSHMEM's nvshmemi_cmp_type):
 * - SIGNAL_CMP_EQ (1) -> NVSHMEM_CMP_EQ (0): Wait until signal == cmp_value
 * - SIGNAL_CMP_NE (2) -> NVSHMEM_CMP_NE (1): Wait until signal != cmp_value
 * - SIGNAL_CMP_GT (3) -> NVSHMEM_CMP_GT (2): Wait until signal > cmp_value
 * - SIGNAL_CMP_GE (4) -> NVSHMEM_CMP_GE (3): Wait until signal >= cmp_value
 * - SIGNAL_CMP_LT (5) -> NVSHMEM_CMP_LT (4): Wait until signal < cmp_value
 * - SIGNAL_CMP_LE (6) -> NVSHMEM_CMP_LE (5): Wait until signal <= cmp_value
 *
 * @param nvshmem_ctx NVSHMEM context with gin_signal_pad
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation (SIGNAL_CMP_EQ, SIGNAL_CMP_GE, etc.)
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ uint64_t nvshmem_signal_wait_until_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int32_t signal_index,
    int32_t cmp,
    uint64_t cmp_value) {
  // Get the GIN signal pad from the context (used for remote atomic signaling)
  TORCH_SYMM_CHECK(
      nvshmem_ctx->gin_signal_pad != nullptr, "gin_signal_pad is null");
  uint64_t* signal_pad = nvshmem_ctx->gin_signal_pad;

  // Calculate the address of the specific signal
  uint64_t* target_signal = signal_pad + signal_index;

  // Map our SIGNAL_CMP_* constants (1-based) to NVSHMEM's native constants
  // (0-based) SIGNAL_CMP_* = NVSHMEM_CMP_* + 1, so we subtract 1
  int nvshmem_cmp = cmp - 1;

  // Use NVSHMEM's signal wait until API
  return nvshmem_signal_wait_until(target_signal, nvshmem_cmp, cmp_value);
}

// =============================================================================
// NVSHMEM TRITON WRAPPERS
// These are the entry points for Triton kernels via extern_elementwise.
// They must be extern "C" to avoid C++ name mangling.
// =============================================================================

extern "C" {

/**
 * NVSHMEM-specific wrapper for all-reduce.
 */
__device__ int32_t nvshmem_symm_all_reduce(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int32_t byte_offset,
    int32_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  float* buffer = reinterpret_cast<float*>(local_ptr);
  nvshmem_all_reduce_impl(
      nvshmem_ctx, buffer, byte_offset, num_elements, reduce_op, dtype);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for quiet operation.
 */
__device__ int32_t nvshmem_symm_quiet(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_quiet_impl(nvshmem_ctx);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for barrier operation.
 */
__device__ int32_t nvshmem_symm_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_barrier_impl(nvshmem_ctx);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for fence operation.
 */
__device__ int32_t nvshmem_symm_fence(int64_t ctx_ptr, int32_t scope) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_fence_impl(nvshmem_ctx, scope);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for remote_ptr operation.
 */
__device__ int64_t
nvshmem_symm_remote_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  return nvshmem_remote_ptr_impl(nvshmem_ctx, local_ptr, peer);
}

/**
 * NVSHMEM-specific wrapper for multicast_ptr operation.
 */
__device__ int64_t nvshmem_symm_multicast_ptr(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t team_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_multicast_ptr_impl(nvshmem_ctx, local_ptr, nvshmem_team);
}

/**
 * NVSHMEM-specific wrapper for signal operation.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * This is a point-to-point notification mechanism without data transfer.
 * Uses the GIN signal pad stored in the context.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination PE to signal
 * @param value Value to use in the operation
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ int32_t nvshmem_symm_signal(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t dest_rank,
    int64_t value,
    int32_t op) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_signal_impl(
      nvshmem_ctx, signal_index, dest_rank, static_cast<uint64_t>(value), op);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for signal_ptr operation.
 *
 * Returns a device pointer to a peer's LSA signal pad, if accessible via P2P.
 * This allows direct load/store access to the peer's signal memory.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param peer Peer PE number to get signal pad pointer for
 * @return Device pointer to peer's LSA signal pad, or 0 if not accessible
 */
__device__ int64_t nvshmem_symm_signal_ptr(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  return nvshmem_signal_ptr_impl(nvshmem_ctx, peer);
}

/**
 * NVSHMEM-specific wrapper for signal_wait_until operation.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Uses the gin_signal_pad from the context (same pad that symm_signal writes
 * to). This enables point-to-point synchronization patterns.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation (SIGNAL_CMP_EQ, SIGNAL_CMP_GE, etc.)
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ int64_t nvshmem_symm_signal_wait_until(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t cmp,
    int64_t cmp_value) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  return static_cast<int64_t>(nvshmem_signal_wait_until_impl(
      nvshmem_ctx, signal_index, cmp, static_cast<uint64_t>(cmp_value)));
}

// =============================================================================
// NVSHMEM TEAM PRIMITIVES
// =============================================================================

/**
 * NVSHMEM-specific wrapper for team_size.
 */
__device__ int32_t nvshmem_symm_team_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_team->team_size;
}

/**
 * NVSHMEM-specific wrapper for team_rank.
 */
__device__ int32_t nvshmem_symm_team_rank(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_team->team_rank;
}

/**
 * NVSHMEM-specific wrapper for team_lsa_size.
 */
__device__ int32_t nvshmem_symm_team_lsa_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_team->lsa_size;
}

/**
 * NVSHMEM-specific wrapper for team_lsa.
 */
__device__ int32_t nvshmem_symm_team_lsa(int64_t team_ptr, int32_t peer) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_team->is_lsa_peer(peer) ? 1 : 0;
}

} // extern "C"
