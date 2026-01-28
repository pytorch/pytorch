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
 * NVSHMEM backend implementation of lsa_barrier.
 *
 * Performs barrier synchronization among ranks in the same LSA (Local Symmetric
 * Access) domain. Only ranks that can directly access each other's memory via
 * load/store operations participate in this barrier.
 *
 * This is useful for synchronizing within a local group (e.g., GPUs on the same
 * node with NVLink connectivity) without waiting for remote ranks.
 *
 * Uses nvshmemx_barrier_block(NVSHMEM_TEAM_SHARED) for synchronization within
 * the LSA domain. NVSHMEM_TEAM_SHARED (value 1) represents PEs that share
 * memory on the same node.
 *
 * @param nvshmem_ctx NVSHMEM context (unused, for API consistency)
 */
__device__ void nvshmem_lsa_barrier_impl(NVSHMEMSymmContext* nvshmem_ctx) {
  // NVSHMEM_TEAM_SHARED (value 1) represents PEs that share memory (same node)
  constexpr int32_t NVSHMEM_TEAM_SHARED = 1;

  // Use nvshmemx_barrier_block with NVSHMEM_TEAM_SHARED for LSA domain barrier
  nvshmemx_barrier_block(NVSHMEM_TEAM_SHARED);
}

/**
 * NVSHMEM backend implementation of lsa_ptr.
 */
__device__ int64_t nvshmem_lsa_ptr_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int64_t local_ptr,
    int32_t peer) {
  void* ptr = reinterpret_cast<void*>(local_ptr);
  void* peer_ptr = nvshmem_ptr(ptr, peer);
  return reinterpret_cast<int64_t>(peer_ptr);
}

/**
 * NVSHMEM backend implementation of lsa_multicast_ptr.
 */
__device__ int64_t nvshmem_lsa_multicast_ptr_impl(
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
 * NVSHMEM backend implementation of lsa_signal_ptr.
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
nvshmem_lsa_signal_ptr_impl(NVSHMEMSymmContext* nvshmem_ctx, int32_t peer) {
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

/**
 * NVSHMEM backend implementation of signal_reset.
 *
 * Resets a local signal at signal_index to zero. This is used to prepare
 * a signal for the next round of signaling/waiting in iterative algorithms.
 *
 * The implementation uses nvshmem_signal_wait_until to ensure the signal
 * has reached the expected value before resetting it to zero. This provides
 * a memory fence and ensures proper ordering.
 *
 * Uses the gin_signal_pad from the context (same pad that symm_signal writes
 * to and symm_signal_wait_until reads from).
 *
 * @param nvshmem_ctx NVSHMEM context with gin_signal_pad
 * @param signal_index Index of the signal to reset
 */
__device__ void nvshmem_signal_reset_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    int32_t signal_index) {
  // Get the GIN signal pad from the context (used for remote atomic signaling)
  TORCH_SYMM_CHECK(
      nvshmem_ctx->gin_signal_pad != nullptr, "gin_signal_pad is null");
  uint64_t* signal_pad = nvshmem_ctx->gin_signal_pad;

  // Calculate the address of the specific signal
  uint64_t* target_signal = signal_pad + signal_index;

  // Wait until the signal is non-zero (ensuring any pending signal has arrived)
  // Using SIGNAL_CMP_NE (2) -> NVSHMEM_CMP_NE (1): Wait until signal != 0
  nvshmem_signal_wait_until(target_signal, 1 /* NVSHMEM_CMP_NE */, 0);

  // Reset the signal to zero
  *target_signal = 0;
}

/**
 * NVSHMEM backend implementation of put_async.
 *
 * Non-blocking one-sided put: copies count elements of element_size bytes
 * from src_ptr (local) to dest_ptr (also a local pointer that maps to
 * destination rank's symmetric buffer). Returns immediately without waiting
 * for completion.
 *
 * The function resolves local pointers to remote pointers using nvshmem_ptr.
 * The dest_ptr argument is treated as a local symmetric address, and this
 * function computes the corresponding address on the destination PE.
 *
 * Uses nvshmem_putmem_nbi (non-blocking immediate) which returns immediately
 * without any synchronization. To ensure completion, call symm_quiet after
 * issuing all put operations.
 *
 * @param nvshmem_ctx NVSHMEM context
 * @param dest_ptr Local pointer that maps to destination (symmetric address)
 * @param src_ptr Local source pointer
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination PE number
 */
__device__ void nvshmem_put_async_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    void* dest_ptr,
    const void* src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank) {
  // Calculate size in bytes
  size_t byte_count =
      static_cast<size_t>(count) * static_cast<size_t>(element_size);

  // Use nvshmem_putmem_nbi for non-blocking immediate put operation
  // dest_ptr is a symmetric address (local pointer into symmetric memory)
  // NVSHMEM will handle the address translation to the remote PE
  // Returns immediately without waiting - use symm_quiet() to ensure completion
  nvshmem_putmem_nbi(dest_ptr, src_ptr, byte_count, dest_rank);
}

/**
 * NVSHMEM backend implementation of put_signal_async.
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
 * Uses nvshmemx_putmem_signal_nbi (non-blocking immediate) which returns
 * immediately without any synchronization. To ensure completion, call
 * symm_quiet after issuing all put operations.
 *
 * @param nvshmem_ctx NVSHMEM context with gin_signal_pad for signaling
 * @param dest_ptr Local pointer that maps to destination (symmetric address)
 * @param src_ptr Local source pointer
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination PE number
 * @param signal_index Index into the signal pad to update on dest_rank
 * @param signal_value Value to use in the signal operation (default=1)
 * @param signal_op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ void nvshmem_put_signal_async_impl(
    NVSHMEMSymmContext* nvshmem_ctx,
    void* dest_ptr,
    const void* src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank,
    int32_t signal_index,
    uint64_t signal_value,
    int32_t signal_op) {
  // Get the GIN signal pad from the context (used for remote atomic signaling)
  TORCH_SYMM_CHECK(
      nvshmem_ctx->gin_signal_pad != nullptr, "gin_signal_pad is null");
  uint64_t* signal_pad = nvshmem_ctx->gin_signal_pad;

  // Calculate the address of the specific signal on dest_rank
  // The signal_pad is symmetric, so we can use it directly with the PE number
  uint64_t* target_signal = signal_pad + signal_index;

  // Calculate size in bytes
  size_t byte_count =
      static_cast<size_t>(count) * static_cast<size_t>(element_size);

  // Map our signal operation constants to NVSHMEM's native constants
  int nvshmem_sig_op;
  switch (signal_op) {
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

  // Use nvshmemx_putmem_signal_block for block-collective put with signal
  // This is a block-collective operation that requires all threads in the block
  // to participate for correct execution.
  // dest_ptr is a symmetric address (local pointer into symmetric memory)
  // NVSHMEM will handle the address translation to the remote PE
  // The signal is updated atomically on dest_rank after data transfer completes
  // Returns immediately without waiting - use symm_quiet() to ensure completion
  nvshmemx_putmem_signal_block(
      dest_ptr,
      src_ptr,
      byte_count,
      target_signal,
      signal_value,
      nvshmem_sig_op,
      dest_rank);
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
 * NVSHMEM-specific wrapper for lsa_barrier operation.
 *
 * Performs barrier synchronization among ranks in the same LSA (Local Symmetric
 * Access) domain. Only ranks that can directly access each other's memory via
 * load/store operations participate in this barrier.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @return 0 on success
 */
__device__ int32_t nvshmem_symm_lsa_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_lsa_barrier_impl(nvshmem_ctx);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for lsa_ptr operation.
 */
__device__ int64_t
nvshmem_symm_lsa_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  return nvshmem_lsa_ptr_impl(nvshmem_ctx, local_ptr, peer);
}

/**
 * NVSHMEM-specific wrapper for lsa_multicast_ptr operation.
 */
__device__ int64_t nvshmem_symm_lsa_multicast_ptr(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t team_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NVSHMEMSymmTeam* nvshmem_team = cast_to_nvshmem_team(team);
  return nvshmem_lsa_multicast_ptr_impl(nvshmem_ctx, local_ptr, nvshmem_team);
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
 * NVSHMEM-specific wrapper for lsa_signal_ptr operation.
 *
 * Returns a device pointer to a peer's LSA signal pad, if accessible via P2P.
 * This allows direct load/store access to the peer's signal memory.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param peer Peer PE number to get signal pad pointer for
 * @return Device pointer to peer's LSA signal pad, or 0 if not accessible
 */
__device__ int64_t nvshmem_symm_lsa_signal_ptr(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  return nvshmem_lsa_signal_ptr_impl(nvshmem_ctx, peer);
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

/**
 * NVSHMEM-specific wrapper for signal_reset operation.
 *
 * Resets a local signal at signal_index to zero. This is used to prepare
 * a signal for the next round of signaling/waiting in iterative algorithms.
 *
 * Uses nvshmem_signal_wait_until to ensure the signal has arrived before
 * resetting it to zero.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to reset
 * @return 0 on success
 */
__device__ int32_t
nvshmem_symm_signal_reset(int64_t ctx_ptr, int32_t signal_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  nvshmem_signal_reset_impl(nvshmem_ctx, signal_index);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for put_async operation.
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
 * @param dest_rank Destination PE number
 * @return 0 on success
 */
__device__ int32_t nvshmem_symm_put_async(
    int64_t ctx_ptr,
    int64_t dest_ptr,
    int64_t src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
  nvshmem_put_async_impl(
      nvshmem_ctx, dest, src, count, element_size, dest_rank);
  return 0;
}

/**
 * NVSHMEM-specific wrapper for put_signal_async operation.
 *
 * Non-blocking one-sided put with remote signal: copies count elements of
 * element_size bytes from src_ptr (local) to dest_ptr (symmetric address) on
 * dest_rank's buffer. After the data transfer completes, atomically updates
 * the remote signal at signal_index on dest_rank using the specified operation.
 *
 * This is a fused operation that combines data transfer with signaling,
 * allowing the receiver to know when the data has arrived without polling.
 *
 * Returns immediately without waiting for completion. Use symm_quiet to
 * ensure all prior put operations have completed.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param dest_ptr Destination pointer (symmetric address, as int64)
 * @param src_ptr Source pointer (local address, as int64)
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination PE number
 * @param signal_index Index into the signal pad to update on dest_rank
 * @param signal_value Value to use in the signal operation
 * @param signal_op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 * @return 0 on success
 */
__device__ int32_t nvshmem_symm_put_signal_async(
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
  NVSHMEMSymmContext* nvshmem_ctx = cast_to_nvshmem_context(ctx);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
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
