// nccl_symm.cuh
// NCCL backend implementation for symmetric memory operations
//
// This file contains NCCL-specific device functions that can be compiled
// independently and linked into the unified torch_symm.bc bitcode library.
//
// Note: NCCL does NOT provide a device bitcode library (libnccl_device.bc).
// This implementation is included for completeness and can be tested for
// compilation errors. When linked with torch_symm.cu, NCCL_HAS_DEVICE_BITCODE
// controls whether NCCL backend dispatch is enabled.
//
// BUILD MODES:
// - TORCH_SYMM_BITCODE_BUILD=1: Bitcode build for Triton - NCCL device code
//   is excluded to avoid unresolved symbols when linking with Triton kernels.
// - Regular PyTorch build: NCCL device code is included when
//   NCCL_SYMM_TYPES_AVAILABLE is defined.
//
// NCCL LSA Barrier API:
// The proper NCCL LSA barrier API uses ncclLsaBarrierSession<Coop> which is
// a RAII class that manages barrier state. It provides:
// - arrive(): Signal arrival at barrier
// - wait(): Wait for all peers to arrive
// - sync(): arrive() + wait() combined
//
// Cooperative group types (Coop):
// - ncclCoopCta: Block-level cooperative group (__syncthreads())
// - ncclCoopWarp: Warp-level cooperative group
// - ncclCoopThread: Single thread
//
// The barrier session is created with:
// - ncclLsaBarrierSession<ncclCoopCta>(ncclCoopCta{}, comm, ncclTeamTagLsa{},
//                                       barrier_index, use_multimem)

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
#ifndef SIGNAL_OP_SET
#define SIGNAL_OP_SET 0
#define SIGNAL_OP_ADD 1
#endif

// Signal comparison condition constants (for symm_signal_wait_until)
// These match the abstracted constants in symm_comm.cuh
#ifndef SIGNAL_CMP_EQ
#define SIGNAL_CMP_EQ 1 // Equal
#define SIGNAL_CMP_NE 2 // Not equal
#define SIGNAL_CMP_GT 3 // Greater than
#define SIGNAL_CMP_GE 4 // Greater than or equal
#define SIGNAL_CMP_LT 5 // Less than
#define SIGNAL_CMP_LE 6 // Less than or equal
#endif

// =============================================================================
// NCCL CAST HELPERS
// =============================================================================

/**
 * Cast SymmContext to NCCLSymmContext after runtime type check.
 * Asserts on failure.
 */
__device__ __forceinline__ NCCLSymmContext* cast_to_nccl_context(
    SymmContext* ctx) {
  TORCH_SYMM_CHECK(ctx != nullptr, "SymmContext is null");
  TORCH_SYMM_CHECK(
      ctx->type == SymmContext::Type::NCCL, "SymmContext is not NCCL type");
  return static_cast<NCCLSymmContext*>(ctx);
}

/**
 * Cast SymmTeam to NCCLSymmTeam after runtime type check.
 * Asserts on failure.
 */
__device__ __forceinline__ NCCLSymmTeam* cast_to_nccl_team(SymmTeam* team) {
  TORCH_SYMM_CHECK(team != nullptr, "SymmTeam is null");
  TORCH_SYMM_CHECK(
      team->type == SymmTeam::Type::NCCL, "SymmTeam is not NCCL type");
  return static_cast<NCCLSymmTeam*>(team);
}

// =============================================================================
// NCCL HELPER FUNCTIONS
// =============================================================================

// Skip NCCL device implementations during bitcode build to avoid unresolved
// symbols when linking with Triton kernels. These will be compiled when
// PyTorch is built normally with NCCL_SYMM_TYPES_AVAILABLE defined.
#if !defined(TORCH_SYMM_BITCODE_BUILD)

/**
 * Get a pointer to a peer's symmetric buffer at a given offset (NCCL).
 */
__device__ __forceinline__ void* nccl_get_peer_ptr(
    NCCLSymmContext* nccl_ctx,
    int peer,
    size_t byte_offset) {
  return ncclGetLsaPointer(nccl_ctx->buffer_window, byte_offset, peer);
}

/**
 * Perform LSA barrier synchronization using ncclLsaBarrierSession.
 *
 * This is the correct NCCL LSA barrier API. The barrier session is a RAII
 * object that manages barrier state and provides arrive/wait/sync operations.
 *
 * @param nccl_ctx NCCL symmetric context with device communicator
 * @param barrier_index Index of the barrier to use (0..nBarriers-1)
 * @param use_multimem Whether to use multicast (NVSwitch) for barrier
 */
__device__ __forceinline__ void nccl_lsa_barrier_sync(
    NCCLSymmContext* nccl_ctx,
    uint32_t barrier_index,
    bool use_multimem = false) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // Create a CTA-scoped barrier session and sync
  // ncclLsaBarrierSession is a RAII class - sync() is called, then destructor
  // updates the epoch state
  ncclLsaBarrierSession<ncclCoopCta> barrier(
      ncclCoopCta{},
      *nccl_ctx->dev_comm,
      ncclTeamTagLsa{},
      barrier_index,
      use_multimem);
  barrier.sync(ncclCoopCta{}, cuda::memory_order_seq_cst);
#endif
}

// =============================================================================
// NCCL BACKEND IMPLEMENTATIONS
// =============================================================================

/**
 * NCCL backend implementation of all-reduce.
 *
 * DEMONSTRATION ONLY: This kernel implementation is intentionally simple and
 * NOT efficient. It is provided solely to demonstrate the symmetric memory
 * abstraction layer API.
 */
__device__ void nccl_all_reduce_impl(
    NCCLSymmContext* nccl_ctx,
    float* local_buffer,
    int64_t byte_offset,
    int64_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  TORCH_SYMM_CHECK(
      reduce_op == REDUCE_OP_SUM, "Only REDUCE_OP_SUM is supported");
  TORCH_SYMM_CHECK(dtype == DTYPE_FLOAT32, "Only DTYPE_FLOAT32 is supported");

  int rank = nccl_ctx->rank;
  int world_size = nccl_ctx->world_size;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Barrier before reading peer data
  nccl_lsa_barrier_sync(nccl_ctx, 0, false);

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

  // Barrier after writing results
  nccl_lsa_barrier_sync(nccl_ctx, 1, false);
}

/**
 * NCCL backend implementation of quiet.
 *
 * For NCCL, quiet is implemented as an LSA barrier to ensure all prior
 * operations are visible to peers.
 */
__device__ void nccl_quiet_impl(NCCLSymmContext* nccl_ctx) {
  nccl_lsa_barrier_sync(nccl_ctx, 0, false);
}

/**
 * NCCL backend implementation of barrier.
 *
 * Uses ncclLsaBarrierSession for proper LSA barrier synchronization.
 */
__device__ void nccl_barrier_impl(
    NCCLSymmContext* nccl_ctx,
    int32_t barrier_id) {
  nccl_lsa_barrier_sync(nccl_ctx, static_cast<uint32_t>(barrier_id), false);
}

/**
 * NCCL backend implementation of fence.
 */
__device__ void nccl_fence_impl(NCCLSymmContext* nccl_ctx, int32_t scope) {
  switch (scope) {
    case FENCE_SCOPE_CTA:
      __syncthreads();
      break;
    case FENCE_SCOPE_GPU:
      __threadfence();
      break;
    case FENCE_SCOPE_SYSTEM:
      __threadfence_system();
      break;
    default:
      TORCH_SYMM_CHECK(false, "Invalid fence scope");
  }
}

/**
 * NCCL backend implementation of lsa_barrier.
 *
 * Performs barrier synchronization among ranks in the same LSA (Local Symmetric
 * Access) domain using ncclLsaBarrierSession with ncclTeamTagLsa.
 *
 * The ncclTeamTagLsa tag selects the LSA team which contains only ranks that
 * can directly access each other's memory via load/store operations.
 *
 * @param nccl_ctx NCCL context with device communicator
 */
__device__ void nccl_lsa_barrier_impl(NCCLSymmContext* nccl_ctx) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // Use ncclLsaBarrierSession with ncclTeamTagLsa for LSA domain barrier
  // This synchronizes only with ranks in the same LSA domain (same node with
  // NVLink connectivity)
  // Barrier index 0 is used for LSA barriers, use_multimem = false
  ncclLsaBarrierSession<ncclCoopCta> barrier(
      ncclCoopCta{},
      *nccl_ctx->dev_comm,
      ncclTeamTagLsa{},
      0, // barrier_index
      false // use_multimem
  );
  barrier.sync(ncclCoopCta{}, cuda::memory_order_seq_cst);
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(
      false, "NCCL lsa_barrier requires NCCL_SYMM_TYPES_AVAILABLE");
#endif
}

/**
 * NCCL backend implementation of lsa_ptr.
 */
__device__ int64_t
nccl_lsa_ptr_impl(NCCLSymmContext* nccl_ctx, int64_t local_ptr, int32_t peer) {
  size_t byte_offset = reinterpret_cast<char*>(local_ptr) -
      reinterpret_cast<char*>(nccl_ctx->local_buffer);
  void* peer_ptr =
      ncclGetLsaPointer(nccl_ctx->buffer_window, byte_offset, peer);
  return reinterpret_cast<int64_t>(peer_ptr);
}

/**
 * NCCL backend implementation of lsa_multicast_ptr.
 */
__device__ int64_t
nccl_lsa_multicast_ptr_impl(NCCLSymmContext* nccl_ctx, int64_t local_ptr) {
  size_t byte_offset = reinterpret_cast<char*>(local_ptr) -
      reinterpret_cast<char*>(nccl_ctx->local_buffer);
  void* mc_ptr = ncclGetLsaMultimemPointer(
      nccl_ctx->buffer_window, byte_offset, nccl_ctx->dev_comm);
  return reinterpret_cast<int64_t>(mc_ptr);
}

/**
 * NCCL backend implementation of signal.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * For NCCL, this uses the ncclGin (GPU-Initiated Networking) API for signaling.
 *
 * The ncclGin::signal API provides direct signaling without data transfer:
 * - ncclGin_SignalInc{signalIndex}: Increment remote signal by 1
 * - ncclGin_SignalAdd{signalIndex, value}: Add arbitrary value to remote signal
 *
 * @param nccl_ctx NCCL context with signal_pad_ptrs and dev_comm
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination rank to signal
 * @param value Value to use in the operation
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ void nccl_signal_impl(
    NCCLSymmContext* nccl_ctx,
    int32_t signal_index,
    int32_t dest_rank,
    uint64_t value,
    int32_t op) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // ncclGin only supports ADD operations (SignalInc and SignalAdd)
  TORCH_SYMM_CHECK(
      op == SIGNAL_OP_ADD, "NCCL signal only supports SIGNAL_OP_ADD");

  // Use ncclGin for GPU-initiated signaling
  // Initialize GIN context (context ID 0 for simplicity)
  int ginContext = 0;
  ncclGin gin{*nccl_ctx->dev_comm, ginContext};

  // Use ncclGin::signal directly:
  // - value == 1: Use ncclGin_SignalInc (increment by 1)
  // - value > 1:  Use ncclGin_SignalAdd (add arbitrary value)
  if (value == 1) {
    // Use ncclGin::signal with SignalInc to atomically increment by 1
    gin.signal(
        ncclTeamWorld(*nccl_ctx->dev_comm),
        dest_rank,
        ncclGin_SignalInc{static_cast<uint32_t>(signal_index)});
  } else {
    // Use ncclGin::signal with SignalAdd for arbitrary value
    gin.signal(
        ncclTeamWorld(*nccl_ctx->dev_comm),
        dest_rank,
        ncclGin_SignalAdd{static_cast<uint32_t>(signal_index), value});
  }
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(false, "NCCL signal requires NCCL_SYMM_TYPES_AVAILABLE");
#endif
}

/**
 * NCCL backend implementation of lsa_signal_ptr.
 *
 * Returns a device pointer to a peer's signal pad, if accessible via P2P.
 * For NCCL, this uses the LSA window to get the peer's signal pad address.
 *
 * @param nccl_ctx NCCL context with signal_window
 * @param peer Peer rank to get signal pad pointer for
 * @return Device pointer to peer's signal pad, or 0 if not accessible
 */
__device__ int64_t
nccl_lsa_signal_ptr_impl(NCCLSymmContext* nccl_ctx, int32_t peer) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // Use ncclGetLsaPointer with signal_window to get peer's signal pad
  // Offset 0 gives the base of the signal pad
  void* peer_signal_pad = ncclGetLsaPointer(nccl_ctx->signal_window, 0, peer);
  return reinterpret_cast<int64_t>(peer_signal_pad);
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(
      false, "NCCL lsa_signal_ptr requires NCCL_SYMM_TYPES_AVAILABLE");
  return 0;
#endif
}

/**
 * NCCL backend implementation of signal_wait_until.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Uses ncclGin::waitSignal for GPU-initiated networking wait operations.
 * Note: NCCL's ncclGin::waitSignal only supports >= condition (wait until
 * signal value meets or exceeds the given threshold).
 *
 * Supported conditions:
 * - SIGNAL_CMP_GE (4): Wait until signal >= cmp_value
 *
 * @param nccl_ctx NCCL context with dev_comm
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation (only SIGNAL_CMP_GE supported for NCCL)
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ uint64_t nccl_signal_wait_until_impl(
    NCCLSymmContext* nccl_ctx,
    int32_t signal_index,
    int32_t cmp,
    uint64_t cmp_value) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // NCCL's ncclGin::waitSignal only supports waiting for >= condition
  // SIGNAL_CMP_GE is our constant value 4
  TORCH_SYMM_CHECK(
      cmp == SIGNAL_CMP_GE,
      "NCCL signal_wait_until only supports SIGNAL_CMP_GE condition");

  // Initialize GIN context (context ID 0 for simplicity)
  int ginContext = 0;
  ncclGin gin{*nccl_ctx->dev_comm, ginContext};

  // Use ncclGin::waitSignal to wait until signal >= cmp_value
  // waitSignal<Coop>(signal, least, bits, ord)
  // We use ncclCoopCta for block-level cooperation
  gin.waitSignal(
      ncclCoopCta{}, static_cast<ncclGinSignal_t>(signal_index), cmp_value);

  // Return the comparison value as NCCL doesn't return the actual signal value
  return cmp_value;
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(
      false, "NCCL signal_wait_until requires NCCL_SYMM_TYPES_AVAILABLE");
  return 0;
#endif
}

// =============================================================================
// NCCL TRITON WRAPPERS
// These are the entry points for Triton kernels via extern_elementwise.
// They must be extern "C" to avoid C++ name mangling.
// =============================================================================

extern "C" {

/**
 * NCCL-specific wrapper for all-reduce.
 */
__device__ int32_t nccl_symm_all_reduce(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t byte_offset,
    int64_t num_elements,
    int32_t reduce_op,
    int32_t dtype) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  float* buffer = reinterpret_cast<float*>(local_ptr);
  nccl_all_reduce_impl(
      nccl_ctx, buffer, byte_offset, num_elements, reduce_op, dtype);
  return 0;
}

/**
 * NCCL-specific wrapper for quiet operation.
 */
__device__ int32_t nccl_symm_quiet(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_quiet_impl(nccl_ctx);
  return 0;
}

/**
 * NCCL-specific wrapper for barrier operation.
 */
__device__ int32_t nccl_symm_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_barrier_impl(nccl_ctx, 0);
  return 0;
}

/**
 * NCCL-specific wrapper for fence operation.
 */
__device__ int32_t nccl_symm_fence(int64_t ctx_ptr, int32_t scope) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_fence_impl(nccl_ctx, scope);
  return 0;
}

/**
 * NCCL-specific wrapper for lsa_barrier operation.
 *
 * Performs barrier synchronization among ranks in the same LSA (Local Symmetric
 * Access) domain. Only ranks that can directly access each other's memory via
 * load/store operations participate in this barrier.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @return 0 on success
 */
__device__ int32_t nccl_symm_lsa_barrier(int64_t ctx_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_lsa_barrier_impl(nccl_ctx);
  return 0;
}

/**
 * NCCL-specific wrapper for lsa_ptr operation.
 */
__device__ int64_t
nccl_symm_lsa_ptr(int64_t ctx_ptr, int64_t local_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  return nccl_lsa_ptr_impl(nccl_ctx, local_ptr, peer);
}

/**
 * NCCL-specific wrapper for lsa_multicast_ptr operation.
 */
__device__ int64_t nccl_symm_lsa_multicast_ptr(
    int64_t ctx_ptr,
    int64_t local_ptr,
    int64_t team_ptr) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  return nccl_lsa_multicast_ptr_impl(nccl_ctx, local_ptr);
}

/**
 * NCCL-specific wrapper for signal operation.
 *
 * Atomically updates a signal value at a remote rank's signal location.
 * Uses the signal pad stored in the context.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to update
 * @param dest_rank Destination rank to signal
 * @param value Value to use in the operation
 * @param op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ int32_t nccl_symm_signal(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t dest_rank,
    int64_t value,
    int32_t op) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_signal_impl(
      nccl_ctx, signal_index, dest_rank, static_cast<uint64_t>(value), op);
  return 0;
}

/**
 * NCCL-specific wrapper for lsa_signal_ptr operation.
 *
 * Returns a device pointer to a peer's signal pad, if accessible via P2P.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param peer Peer rank to get signal pad pointer for
 * @return Device pointer to peer's signal pad, or 0 if not accessible
 */
__device__ int64_t nccl_symm_lsa_signal_ptr(int64_t ctx_ptr, int32_t peer) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  return nccl_lsa_signal_ptr_impl(nccl_ctx, peer);
}

/**
 * NCCL-specific wrapper for signal_wait_until operation.
 *
 * Blocks the calling thread/CTA until a local signal at signal_index meets
 * the specified condition relative to the comparison value.
 *
 * Uses ncclGin::waitSignal for GPU-initiated networking wait operations.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to wait on
 * @param cmp Comparison operation (only SIGNAL_CMP_GE supported for NCCL)
 * @param cmp_value Value to compare against
 * @return The signal value that satisfied the condition
 */
__device__ int64_t nccl_symm_signal_wait_until(
    int64_t ctx_ptr,
    int32_t signal_index,
    int32_t cmp,
    int64_t cmp_value) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  return static_cast<int64_t>(nccl_signal_wait_until_impl(
      nccl_ctx, signal_index, cmp, static_cast<uint64_t>(cmp_value)));
}

/**
 * NCCL backend implementation of signal_reset.
 *
 * Resets a local signal at signal_index to zero. This is used to prepare
 * a signal for the next round of signaling/waiting in iterative algorithms.
 *
 * Uses ncclGin::resetSignal to atomically reset the signal value.
 *
 * @param nccl_ctx NCCL context with dev_comm
 * @param signal_index Index of the signal to reset
 */
__device__ void nccl_signal_reset_impl(
    NCCLSymmContext* nccl_ctx,
    int32_t signal_index) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // Initialize GIN context (context ID 0 for simplicity)
  int ginContext = 0;
  ncclGin gin{*nccl_ctx->dev_comm, ginContext};

  // Use ncclGin::resetSignal to reset the signal value to zero
  gin.resetSignal(static_cast<ncclGinSignal_t>(signal_index));
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(
      false, "NCCL signal_reset requires NCCL_SYMM_TYPES_AVAILABLE");
#endif
}

/**
 * NCCL-specific wrapper for signal_reset operation.
 *
 * Resets a local signal at signal_index to zero. This is used to prepare
 * a signal for the next round of signaling/waiting in iterative algorithms.
 *
 * Uses ncclGin::resetSignal to atomically reset the signal value.
 *
 * @param ctx_ptr Pointer to SymmContext (as int64)
 * @param signal_index Index of the signal to reset
 * @return 0 on success
 */
__device__ int32_t
nccl_symm_signal_reset(int64_t ctx_ptr, int32_t signal_index) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  nccl_signal_reset_impl(nccl_ctx, signal_index);
  return 0;
}

/**
 * NCCL backend implementation of put_async.
 *
 * Non-blocking one-sided put: copies count elements of element_size bytes
 * from src_ptr (local) to dest_ptr (also a local pointer that maps to
 * destination rank's buffer). Returns immediately without waiting for
 * completion.
 *
 * The function uses NCCLSymmContext::resolve_window() to resolve both source
 * and destination pointers to their corresponding ncclWindow_t handles and
 * byte offsets. This allows the GIN put operation to work with any registered
 * memory region.
 *
 * To ensure completion, call symm_quiet after issuing all put operations.
 *
 * @param nccl_ctx NCCL context with window registry and dev_comm
 * @param dest_ptr Local pointer that maps to destination (symmetric address)
 * @param src_ptr Local source pointer
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination rank number
 */
__device__ void nccl_put_async_impl(
    NCCLSymmContext* nccl_ctx,
    void* dest_ptr,
    const void* src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // Resolve destination pointer to window + offset
  NCCLWindowResolution dest_res = nccl_ctx->resolve_window(dest_ptr);
  TORCH_SYMM_CHECK(
      dest_res.valid, "dest_ptr not found in any registered NCCL window");

  // Resolve source pointer to window + offset
  NCCLWindowResolution src_res = nccl_ctx->resolve_window(src_ptr);
  TORCH_SYMM_CHECK(
      src_res.valid, "src_ptr not found in any registered NCCL window");

  // Calculate size in bytes
  size_t byte_count =
      static_cast<size_t>(count) * static_cast<size_t>(element_size);

  // Initialize GIN context (context ID 0 for simplicity)
  int ginContext = 0;
  ncclGin gin{*nccl_ctx->dev_comm, ginContext};

  // Use ncclGin::put with resolved windows and offsets for the data transfer
  // The put operation is non-blocking and returns immediately
  // Args: team, dest_rank, dest_window, dest_offset, src_window, src_offset,
  // byte_count
  gin.put(
      ncclTeamWorld(*nccl_ctx->dev_comm),
      dest_rank,
      dest_res.window,
      dest_res.offset,
      src_res.window,
      src_res.offset,
      byte_count);
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(false, "NCCL put_async requires NCCL_SYMM_TYPES_AVAILABLE");
#endif
}

/**
 * NCCL backend implementation of put_signal_async.
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
 * Uses ncclGin::put followed by ncclGin::signal with appropriate remote action.
 *
 * To ensure completion, call symm_quiet after issuing all put operations.
 *
 * @param nccl_ctx NCCL context with window registry and dev_comm
 * @param dest_ptr Local pointer that maps to destination (symmetric address)
 * @param src_ptr Local source pointer
 * @param count Number of elements to transfer
 * @param element_size Size of each element in bytes
 * @param dest_rank Destination rank number
 * @param signal_index Index into the signal pad to update on dest_rank
 * @param signal_value Value to use in the signal operation (default=1)
 * @param signal_op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 */
__device__ void nccl_put_signal_async_impl(
    NCCLSymmContext* nccl_ctx,
    void* dest_ptr,
    const void* src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank,
    int32_t signal_index,
    uint64_t signal_value,
    int32_t signal_op) {
#if defined(NCCL_SYMM_TYPES_AVAILABLE)
  // NCCL only supports ADD operations for signaling
  TORCH_SYMM_CHECK(
      signal_op == SIGNAL_OP_ADD,
      "NCCL put_signal_async only supports SIGNAL_OP_ADD");

  // Resolve destination pointer to window + offset
  NCCLWindowResolution dest_res = nccl_ctx->resolve_window(dest_ptr);
  TORCH_SYMM_CHECK(
      dest_res.valid, "dest_ptr not found in any registered NCCL window");

  // Resolve source pointer to window + offset
  NCCLWindowResolution src_res = nccl_ctx->resolve_window(src_ptr);
  TORCH_SYMM_CHECK(
      src_res.valid, "src_ptr not found in any registered NCCL window");

  // Calculate size in bytes
  size_t byte_count =
      static_cast<size_t>(count) * static_cast<size_t>(element_size);

  // Initialize GIN context (context ID 0 for simplicity)
  int ginContext = 0;
  ncclGin gin{*nccl_ctx->dev_comm, ginContext};

  // Use ncclGin::put with remote action to atomically update signal after data
  // transfer
  // ncclGin_SignalAdd provides the remote action that adds to the signal
  // The signal is updated atomically on dest_rank after data transfer completes
  if (signal_value == 1) {
    // Use ncclGin_SignalInc for incrementing by 1
    gin.put(
        ncclTeamWorld(*nccl_ctx->dev_comm),
        dest_rank,
        dest_res.window,
        dest_res.offset,
        src_res.window,
        src_res.offset,
        byte_count,
        ncclGin_SignalInc{static_cast<uint32_t>(signal_index)});
  } else {
    // Use ncclGin_SignalAdd for adding arbitrary value
    gin.put(
        ncclTeamWorld(*nccl_ctx->dev_comm),
        dest_rank,
        dest_res.window,
        dest_res.offset,
        src_res.window,
        src_res.offset,
        byte_count,
        ncclGin_SignalAdd{static_cast<uint32_t>(signal_index), signal_value});
  }
#else
  // NCCL device types are not available
  TORCH_SYMM_CHECK(
      false, "NCCL put_signal_async requires NCCL_SYMM_TYPES_AVAILABLE");
#endif
}

/**
 * NCCL-specific wrapper for put_async operation.
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
 * @param dest_rank Destination rank number
 * @return 0 on success
 */
__device__ int32_t nccl_symm_put_async(
    int64_t ctx_ptr,
    int64_t dest_ptr,
    int64_t src_ptr,
    int32_t count,
    int32_t element_size,
    int32_t dest_rank) {
  SymmContext* ctx = reinterpret_cast<SymmContext*>(ctx_ptr);
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
  nccl_put_async_impl(nccl_ctx, dest, src, count, element_size, dest_rank);
  return 0;
}

/**
 * NCCL-specific wrapper for put_signal_async operation.
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
 * @param dest_rank Destination rank number
 * @param signal_index Index into the signal pad to update on dest_rank
 * @param signal_value Value to use in the signal operation
 * @param signal_op Signal operation: SIGNAL_OP_SET (0) or SIGNAL_OP_ADD (1)
 * @return 0 on success
 */
__device__ int32_t nccl_symm_put_signal_async(
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
  NCCLSymmContext* nccl_ctx = cast_to_nccl_context(ctx);
  void* dest = reinterpret_cast<void*>(dest_ptr);
  const void* src = reinterpret_cast<const void*>(src_ptr);
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

// =============================================================================
// NCCL TEAM PRIMITIVES
// =============================================================================

/**
 * NCCL-specific wrapper for team_size.
 */
__device__ int32_t nccl_symm_team_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NCCLSymmTeam* nccl_team = cast_to_nccl_team(team);
  return nccl_team->team_size;
}

/**
 * NCCL-specific wrapper for team_rank.
 */
__device__ int32_t nccl_symm_team_rank(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NCCLSymmTeam* nccl_team = cast_to_nccl_team(team);
  return nccl_team->team_rank;
}

/**
 * NCCL-specific wrapper for team_lsa_size.
 */
__device__ int32_t nccl_symm_team_lsa_size(int64_t team_ptr) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NCCLSymmTeam* nccl_team = cast_to_nccl_team(team);
  return nccl_team->lsa_size;
}

/**
 * NCCL-specific wrapper for team_lsa.
 */
__device__ int32_t nccl_symm_team_lsa(int64_t team_ptr, int32_t peer) {
  SymmTeam* team = reinterpret_cast<SymmTeam*>(team_ptr);
  NCCLSymmTeam* nccl_team = cast_to_nccl_team(team);
  return nccl_team->is_lsa_peer(peer) ? 1 : 0;
}

} // extern "C"

#endif // !defined(TORCH_SYMM_BITCODE_BUILD)
