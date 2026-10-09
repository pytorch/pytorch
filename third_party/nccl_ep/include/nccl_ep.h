/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda.h>
#include <nccl.h>
#include "ep_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

// Library release version. These are the single source of truth — the
// build system (CMake + Makefile) parses them from this header to set the
// shared library's VERSION/SOVERSION (libnccl_ep.so.MAJOR.MINOR.PATCH with
// a libnccl_ep.so.MAJOR soname symlink).
#define NCCL_EP_MAJOR 0
#define NCCL_EP_MINOR 0
#define NCCL_EP_PATCH 1

// Packed version code: MAJOR*10000 + MINOR*100 + PATCH. Mirrors NCCL_VERSION_CODE.
#define NCCL_EP_VERSION_CODE (NCCL_EP_MAJOR * 10000 + NCCL_EP_MINOR * 100 + NCCL_EP_PATCH)

// ============================================================================
// ABI + API versioning
//
// Every struct that crosses the API boundary starts with `unsigned int size`,
// which the caller MUST set to sizeof(struct). The library validates it
// against its own sizeof and rejects mismatches (layout / ABI check).
//
// Immediately after `size`, every struct carries an
// `unsigned int magic` field, which the caller MUST set to a predefined value.
// The library rejects structs whose magic does not match — catching
// uninitialised / zero-filled / wrong-type pointers at the API boundary.
//
// ncclEpGroupConfig_t additionally carries `unsigned int version`
// (= NCCL_EP_API_VERSION) for catching feature-level incompatibilities that
// size can't detect; the library warns on mismatch.
//
// Convenience macros NCCL_EP_xxx_INIT expand to compound literals that pre-fill
// these fields. They work in declaration init, assignment, and expression
// contexts:
//
//   ncclEpDispatchInputs_t inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
//   inputs.tokens = my_tokens;
//
//   // also valid as a post-declaration assignment / reset:
//   inputs = NCCL_EP_DISPATCH_INPUTS_INIT;
//
// To set additional fields inline, write the compound literal directly:
//
//   inputs = (ncclEpDispatchInputs_t){
//       .size   = sizeof(ncclEpDispatchInputs_t),
//       .magic  = NCCL_EP_MAGIC,
//       .tokens = my_tokens,
//   };
//
// FUTURE IMPROVEMENT: relax the strict size-equality check to allow forward
// compat when the caller's struct is larger than the library's known size, by
// scanning the trailing bytes; if all zero, accept silently (the caller didn't
// actually fill any unknown fields).
// ============================================================================
#define NCCL_EP_API_VERSION 1
#define NCCL_EP_MAGIC 0xC00FFFEEu

// Return the NCCL_EP_VERSION_CODE of the NCCL EP library in the supplied integer.
// This integer is coded with the MAJOR, MINOR and PATCH level of the library.
//
// Arguments:
//   version - [OUT] Pointer to receive the library's NCCL_EP_VERSION_CODE value
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetVersion(int* version);

// N-dimensional tensor descriptor — a lightweight structure
// that can be both allocated by the user and by the NCCL EP library.
//
// Users can declare inline on the stack, as a struct member, or as a global/static object.
// No heap allocation is needed for the descriptor itself.
// In this case, the caller owns the device buffer (data pointer) or NCCL window registration,
// and also the `sizes` array that describes the per-dimension
// extents. Both must outlive any library call that observes this descriptor.
//
// Always initialise with NCCL_EP_TENSOR_INIT then fill fields directly:
//   size_t dims[2] = { N, H };
//   ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
//   t.ndim = 2; t.datatype = ncclFloat16; t.data = data_ptr; t.sizes = dims;
//
// When all user fields are known up-front, NCCL_EP_TENSOR_INIT_INLINE provides
// the service-field designators for in-place construction via a compound literal:
//
//   size_t dims[2] = { N, H };
//   ncclEpTensor_t t = { NCCL_EP_TENSOR_INIT_INLINE,
//                        .ndim = 2, .datatype = ncclFloat16,
//                        .data = data_ptr, .sizes = dims };
//
// No explicit destruction is needed — the descriptor holds no resources.
//
// NOTE: for the library-allocated tensors, the caller can use ncclEpTensorAlloc.

#define NCCL_EP_TENSOR_MAGIC 0xCAFECAFE
typedef struct ncclEpTensor {
    // NOTE: these fields for internal use only
    unsigned int size;          // = sizeof(ncclEpTensor_t); set by NCCL_EP_TENSOR_INIT
    unsigned int magic;         // Init cookie: NCCL_EP_TENSOR_MAGIC (user-initialised)
                                // or an internal DYNAMIC value (ncclEpTensorAlloc).
                                // 0 = uninitialised; the library rejects such tensors.

    // Fields are set by the user
    unsigned int ndim;          // number of dimensions
    ncclDataType_t datatype;
    void* data;                 // device pointer (NULL for window-backed tensors until resolved)
    ncclWindow_t win_hdl;       // NCCL window handle (NULL for plain device-pointer tensors)
    uint64_t win_offset;        // byte offset within win_hdl
    size_t* sizes;              // caller-owned array of length `ndim`, per-dimension sizes
} ncclEpTensor_t;

// Static tensor initializer
#define NCCL_EP_TENSOR_INIT_INLINE \
    .size  = (unsigned int)sizeof(ncclEpTensor_t), \
    .magic = NCCL_EP_TENSOR_MAGIC
#define NCCL_EP_TENSOR_INIT ((ncclEpTensor_t){ NCCL_EP_TENSOR_INIT_INLINE })

// Allocation configuration for future extensions of ncclEpTensorAlloc.
// Callers should either pass NULL (defaults) or initialise via
// NCCL_EP_TENSOR_ALLOC_CONFIG_INIT.
typedef struct {
    unsigned int size;  // = sizeof(this struct); first field, never moves
    unsigned int magic; // = NCCL_EP_MAGIC; second field, never moves
} ncclEpTensorAllocConfig_t;

#define NCCL_EP_TENSOR_ALLOC_CONFIG_INIT \
    ((ncclEpTensorAllocConfig_t){ \
        .size  = (unsigned int)sizeof(ncclEpTensorAllocConfig_t), \
        .magic = NCCL_EP_MAGIC })

// Allocate a tensor descriptor sufficient to represent the requested shape.
//
// Arguments:
//   tensor   - [OUT] On success, receives a pointer to the new descriptor.
//   ndim     - [IN]  Number of dimensions (> 0).
//   datatype - [IN]  Element type.
//   sizes    - [IN]  Array of `ndim` dimension sizes.
//   config   - [IN]  Optional allocation configuration. NULL = defaults.
//
// Returns: ncclResult_t error code.
ncclResult_t ncclEpTensorAlloc(
    ncclEpTensor_t** tensor,
    unsigned int ndim,
    ncclDataType_t datatype,
    const size_t* sizes,
    const ncclEpTensorAllocConfig_t* config
);

// Release a descriptor previously returned by ncclEpTensorAlloc.
//
// Arguments:
//   tensor - [IN] Pointer returned by ncclEpTensorAlloc. NULL is accepted.
//
// Returns: ncclResult_t error code.
ncclResult_t ncclEpTensorDestroy(
    ncclEpTensor_t* tensor
);

// Allocator and free function pointer types.
// context is the value stored in ncclEpAllocConfig_t::context and is forwarded unchanged
// on every call.
typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size, void* context);
typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr, void* context);

// Device memory allocator configuration embedded in ncclEpGroupConfig_t.
typedef struct {
    // Optional custom device memory allocator (NULL for default cudaMalloc).
    ncclEpAllocFn_t alloc_fn;
    // Optional custom device memory free function (NULL for default cudaFree).
    ncclEpFreeFn_t free_fn;
    // Opaque pointer forwarded verbatim to every alloc_fn/free_fn call.
    void* context;
} ncclEpAllocConfig_t;

// EP group configuration structure
typedef struct {
    unsigned int size;                   // = sizeof(this struct); first field, never moves
    unsigned int magic;                  // = NCCL_EP_MAGIC; second field, never moves
    unsigned int version;                // = NCCL_EP_API_VERSION; caller's feature-set version
    ncclEpAlgorithm_t algorithm;         // low_latency or high_throughput
    unsigned int num_experts;            // Number of experts (required)
    // Maximum number of tokens any single rank will dispatch. Must be the same across all ranks.
    // REQUIRED for both LL and HT modes (must be > 0).
    unsigned int max_dispatch_tokens_per_rank;
    // Maximum number of tokens any single rank will receive.
    //   HT: required, must be >= max_dispatch_tokens_per_rank. If you use the
    //   Expert-Major layout, this size must account for the possibility of
    //   duplicating a token to multiple local experts.
    //   LL: AUTO/0 → nRanks * max_dispatch_tokens_per_rank.
    unsigned int max_recv_tokens_per_rank;
    // Upper bound on per-token bytes, covering both dispatch and combine.
    // The group sizes all token buffers from this; per-call sizes flow through
    // the input tensors' sizes/datatype and may be smaller. Independent of
    // element type — purely byte-oriented.
    unsigned int max_token_bytes;
    // RDMA buffer size in bytes for LL mode. Two modes:
    //   - NCCL_EP_AUTO: the library automatically selects the internal
    //     staging buffer size. The buffer allocation/re-allocation  may be
    //     performed lazily in ncclEpInitHandle once all relevant sizing 
    //     information is known.
    //   - Explicit positive value: the library allocates exactly that
    //     many bytes at ncclEpCreateGroup time. ncclEpInitHandle returns
    //     ncclInvalidUsage if the requested layout doesn't fit; no
    //     reallocation is ever performed.
    //     (NOTE: for optimal sizing, a way to query the required size is planned for future release)
    unsigned long int rdma_buffer_size;
    unsigned int num_qp_per_rank;        // Number of QPs per rank (NCCL_EP_AUTO for auto)
    // Number of channels per rank (NCCL_EP_AUTO for auto).
    // In high throughput collectives, each channel occupies 2 SMs
    unsigned int num_channels;
    // Maximum number of SMs to use for EP kernels (dispatch, combine, preprocessing).
    // Default: NCCL_EP_AUTO — algorithm-dependent default.
    unsigned int max_num_sms;
    // Device memory allocator; zero-init (all NULL) uses cudaMalloc/cudaFree.
    ncclEpAllocConfig_t alloc;
    // Enable active-mask support for fault tolerance (LL mode only).
    // When enabled, a per-rank mask buffer is allocated. If a remote rank times out
    // during dispatch or combine, it is automatically masked (skipped) rather than
    // causing a GPU trap. The mask can be queried, updated, and cleared via the
    // ncclEpMaskQuery / ncclEpMaskUpdate / ncclEpMaskClean APIs.
    // A host-visible error flag is also set on timeout, pollable via ncclEpGetAsyncError().
    unsigned int enable_mask;
    // Timeout for GPU-side wait loops, in nanoseconds. 0 = use default (~100 s).
    // Can be overridden by the NCCL_EP_TIMEOUT_MS environment variable.
    // Setting too low risks false positives (slow ranks marked as failed).
    uint64_t timeout_ns;
} ncclEpGroupConfig_t;

#define NCCL_EP_GROUP_CONFIG_INIT ((ncclEpGroupConfig_t){ \
    .size    = (unsigned int)sizeof(ncclEpGroupConfig_t), \
    .magic   = NCCL_EP_MAGIC, \
    .version = NCCL_EP_API_VERSION })

// Opaque type forward declaration
typedef struct ncclEpGroup* ncclEpGroup_t;

// Create an EP group from an NCCL communicator
//   This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   ep_group   - [OUT] Pointer to newly created EP group
//   comm       - [IN]  Existing NCCL communicator
//   config     - [IN]  Pointer to EP configuration structure.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCreateGroup(
    ncclEpGroup_t* ep_group,
    ncclComm_t comm,
    const ncclEpGroupConfig_t* config
);

// Destroy an EP group and release associated resources.
//
// Arguments:
//   ep_group     - [IN]  EP group to destroy
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpGroupDestroy(
    ncclEpGroup_t ep_group
);

// Layout info passed to ncclEpCreateHandle / ncclEpUpdateHandle and ncclEpDispatch.
// All fields are optional (NULL = not provided). Each field is a pointer to a
// caller-owned descriptor (stack/static/struct-embedded or from ncclEpTensorAlloc).
typedef struct {
    unsigned int    size;                // = sizeof(this struct); first field, never moves
    unsigned int    magic;               // = NCCL_EP_MAGIC; second field, never moves
    ncclEpTensor_t* expert_counters;     // 1D [num_local_experts] int32 (or int64 for HT EM)
                                         //   HT (handle time): per-expert recv counts. Flat: unpadded int32.
                                         //                     EM: padded counts (sum equals output slot count).
                                         //   LL expert-major: per-expert received token counts (dispatch time).
    ncclEpTensor_t* src_rank_counters;   // 1D [num_ranks] int32
                                         //   LL rank-major only: per-source-rank token counts (dispatch time).
    ncclEpTensor_t* expert_offsets;      // 1D [num_local_experts] int32 or int64
                                         //   HT expert-major only: prefix sum of padded per-expert counts.
    ncclEpTensor_t* recv_total_counter;  // 1D [1] int32 or int64
                                         //   HT: scalar total recv token count. Flat: unpadded. EM: padded slot total.
} ncclEpLayoutInfo_t;

#define NCCL_EP_LAYOUT_INFO_INIT ((ncclEpLayoutInfo_t){ \
    .size  = (unsigned int)sizeof(ncclEpLayoutInfo_t), \
    .magic = NCCL_EP_MAGIC })

// Input tensors for ncclEpDispatch.
// All fields except tokens are optional (NULL = not provided). Each field is a
// pointer to a caller-owned descriptor.
typedef struct {
    unsigned int    size;         // = sizeof(this struct); first field, never moves
    unsigned int    magic;        // = NCCL_EP_MAGIC; second field, never moves
    ncclEpTensor_t* tokens;       // required; 2D [num_tokens, hidden]
    ncclEpTensor_t* topk_weights; // optional; 2D [num_tokens, top_k], ncclFloat32
                                  //   LL rank-major: per-token routing weights
                                  //   HT forward: routing weights (topk_idx taken from handle)
    ncclEpTensor_t* scales;       // optional; HT FP8 only; 2D [num_tokens, hidden/128], ncclFloat32
} ncclEpDispatchInputs_t;

#define NCCL_EP_DISPATCH_INPUTS_INIT ((ncclEpDispatchInputs_t){ \
    .size  = (unsigned int)sizeof(ncclEpDispatchInputs_t), \
    .magic = NCCL_EP_MAGIC })

// Output tensors for ncclEpDispatch.
// All fields except tokens are optional (NULL = not provided). Each field is a
// pointer to a caller-owned descriptor.
typedef struct {
    unsigned int    size;         // = sizeof(this struct); first field, never moves
    unsigned int    magic;        // = NCCL_EP_MAGIC; second field, never moves
    ncclEpTensor_t* tokens;       // required; received tokens
    ncclEpTensor_t* topk_weights; // optional; LL rank-major or HT: received top-k weights
                                  //   LL rank-major: ncclFloat32 [num_ranks, max_dispatch_tokens_per_rank, top_k]
    ncclEpTensor_t* scales;       // optional; FP8 only; received per-token scaling factors
    ncclEpTensor_t* topk_idx;     // optional; LL rank-major or HT FLAT: received top-k expert indices
                                  //   Current dispatch-output semantics:
                                  //     HT FLAT: int64 [N(r), top_k], local expert indices on
                                  //       this rank, unused entries are -1.
                                  //     LL rank-major: int32 [num_ranks, max_dispatch_tokens_per_rank, top_k],
                                  //       preserving source top-k order; local expert index on this
                                  //       rank, or -1 when that top-k entry is not hosted locally.
} ncclEpDispatchOutputs_t;

#define NCCL_EP_DISPATCH_OUTPUTS_INIT ((ncclEpDispatchOutputs_t){ \
    .size  = (unsigned int)sizeof(ncclEpDispatchOutputs_t), \
    .magic = NCCL_EP_MAGIC })

// Input tensors for ncclEpCombine.
// All fields except tokens are optional (NULL = not provided). Each field is a
// pointer to a caller-owned descriptor.
typedef struct {
    unsigned int    size;         // = sizeof(this struct); first field, never moves
    unsigned int    magic;        // = NCCL_EP_MAGIC; second field, never moves
    ncclEpTensor_t* tokens;       // required; post-expert activation tensor
    ncclEpTensor_t* topk_weights; // optional; HT backward combine only:
                                  //   2D [num_recv_tokens, top_k], ncclFloat32
} ncclEpCombineInputs_t;

#define NCCL_EP_COMBINE_INPUTS_INIT ((ncclEpCombineInputs_t){ \
    .size  = (unsigned int)sizeof(ncclEpCombineInputs_t), \
    .magic = NCCL_EP_MAGIC })

// Output tensors for ncclEpCombine.
// All fields except tokens are optional (NULL = not provided). Each field is a
// pointer to a caller-owned descriptor.
typedef struct {
    unsigned int    size;         // = sizeof(this struct); first field, never moves
    unsigned int    magic;        // = NCCL_EP_MAGIC; second field, never moves
    ncclEpTensor_t* tokens;       // required; combined output in original token order
    ncclEpTensor_t* topk_weights; // optional; 2D [num_tokens, top_k], ncclFloat32
                                  //   LL expert-major: per-token routing weights applied on receive side
                                  //   HT backward: combined routing weights output
} ncclEpCombineOutputs_t;

#define NCCL_EP_COMBINE_OUTPUTS_INIT ((ncclEpCombineOutputs_t){ \
    .size  = (unsigned int)sizeof(ncclEpCombineOutputs_t), \
    .magic = NCCL_EP_MAGIC })

// Opaque type forward declaration
typedef struct ncclEpHandle* ncclEpHandle_t;
typedef struct {
    unsigned int size;  // = sizeof(this struct); first field, never moves
    unsigned int magic; // = NCCL_EP_MAGIC; second field, never moves
    // HT expert-major only: per-expert zone alignment in tokens (pow2; 0/1 = no padding).
    // Padded slots are zero-filled by dispatch.
    size_t dispatch_output_per_expert_alignment;
} ncclEpHandleConfig_t;

#define NCCL_EP_HANDLE_CONFIG_INIT ((ncclEpHandleConfig_t){ \
    .size  = (unsigned int)sizeof(ncclEpHandleConfig_t), \
    .magic = NCCL_EP_MAGIC })


// Query the device bytes required for a handle's routing buffers.
//
// Arguments:
//   ep_group  - [IN]  A valid EP group
//   layout    - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config    - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   size_out  - [OUT] Required bytes for handle_mem
//   num_topk  - [IN]  Required for LL (> 0); optional for HT
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleMemSize(
    ncclEpGroup_t               ep_group,
    ncclEpLayout_t              layout,
    const ncclEpHandleConfig_t* config,
    size_t*                     size_out,
    int                         num_topk
);

// Create and initialize EP handle.
// Must be called before the first ncclEpDispatch/ncclEpCombine.
//
// NOTE: the impact of auto-sizing of internal buffers (rdma_buffer_size == NCCL_EP_AUTO):
// * collective behaviour: This function MAY be collective and must be called by all ranks
//   in the group in the same order.
// * memory allocation/re-allocation: This function may perform allocation/re-allocation
//   if the new layout/topK requires more memory than the current allocation. All ranks
//   must call this function in lockstep with the same (layout, num_topk).
//   No active communication is allowed between the ranks during this operation.
// * CUDA graph invalidation: This function must not be included in CUDA graph captures.
//
// Use an explicit, sufficiently large rdma_buffer_size if you need to
// avoid collective allocation/re-allocation, mid-stream reallocation, or graph
// invalidation.
//
// handle_mem == NULL:  NCCL EP allocates via alloc_fn; handle owns the memory.
// handle_mem != NULL:  wraps caller-owned 1D ncclUint8 tensor (>= ncclEpHandleMemSize);
//                      handle owns no memory; ncclEpHandleDestroy frees only the struct.
//
// Arguments:
//   handle     - [OUT] Newly created handle
//   ep_group   - [IN]  A valid EP group
//   layout     - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//   config     - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   num_topk   - [IN]  Required for LL (> 0); pass -1 for HT
//   handle_mem - [IN]  NULL = internal alloc; non-NULL = caller-owned device buffer
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpInitHandle(
    ncclEpHandle_t*             handle,
    ncclEpGroup_t               ep_group,
    ncclEpLayout_t              layout,
    const ncclEpHandleConfig_t* config,
    int                         num_topk,
    const ncclEpTensor_t*       handle_mem  // NULL = library allocates internally
);

// Per-step collective: prepare the handle for the given top-k routing decisions.
// Must be called after ncclEpInitHandle and before ncclEpDispatch.
//
// Arguments:
//   handle             - [IN]  Handle from ncclEpInitHandle
//   topk_idx           - [IN]  [num_tokens, top_k] int64
//   layout_info      - [IN/OUT, optional] Named local tensors (NULL = none provided).
//                         HT: layout_info->expert_counters is required when
//                         max_dispatch_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: must be NULL.
//   stream             - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpUpdateHandle(
    ncclEpHandle_t handle,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    cudaStream_t stream
);

// Create, initialize and bind an EP handle to a given layout and routing decisions.
// Must be called before the first ncclEpDispatch/ncclEpCombine.
// Combines the functionality of ncclEpInitHandle and ncclEpUpdateHandle (see above)
//
// Arguments:
//   handle              - [OUT] Pointer to newly created and initialized EP handle
//   ep_group            - [IN]  A valid EP group
//   layout              - [IN]  Receive buffer layout. Required; must not be NCCL_EP_LAYOUT_UNSET.
//                                HT supports FLAT / EXPERT_MAJOR; LL supports EXPERT_MAJOR / RANK_MAJOR.
//   topk_idx            - [IN]  Tensor holding top-K expert indices (routing information)
//   layout_info         - [IN/OUT, optional] Layout info (see ncclEpLayoutInfo_t). NULL = none.
//                         HT: set expert_counters when max_dispatch_tokens_per_rank is NCCL_EP_AUTO.
//                         LL mode: must be NULL.
//   config              - [IN]  Handle configuration (see ncclEpHandleConfig_t). NULL = defaults.
//   stream              - [IN]  CUDA stream
//
// Notes:
//   - If max_dispatch_tokens_per_rank in ncclEpGroupConfig_t was set to NCCL_EP_AUTO,
//     this call may block as the host allocates memory for the actual number
//     of received tokens.
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpCreateHandle(
    ncclEpHandle_t* handle,
    ncclEpGroup_t ep_group,
    ncclEpLayout_t layout,
    const ncclEpTensor_t* topk_idx,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    const ncclEpHandleConfig_t* config,  // NULL = defaults
    cudaStream_t stream
);

// Destroy an EP handle and release all associated resources.
//
// Arguments:
//   handle         - [IN]  EP handle to destroy
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpHandleDestroy(
    ncclEpHandle_t handle
);


// EP dispatch configuration structure
typedef struct {
    unsigned int size;         // = sizeof(this struct); first field, never moves
    unsigned int magic;        // = NCCL_EP_MAGIC; second field, never moves
    unsigned int send_only;    // if non-zero, only initiate transfers; requires ncclEpComplete() afterward
                               //   supported for LL mode only; output tensors must still be preallocated
    unsigned int round_scales; // whether to round the scaling factors tensor into a power of 2
    ncclEpPassDir_t pass_direction; // forward (default) or backward pass; HT-only.
                               //   FWD requires inputs->topk_weights; BWD forbids it and forbids
                               //   outputs->topk_weights / outputs->topk_idx.
} ncclEpDispatchConfig_t;

#define NCCL_EP_DISPATCH_CONFIG_INIT ((ncclEpDispatchConfig_t){ \
    .size  = (unsigned int)sizeof(ncclEpDispatchConfig_t), \
    .magic = NCCL_EP_MAGIC })

// Perform EP dispatch
//   * Sends tokens and metadata to the experts according to routing decisions.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle        - [IN,OUT] EP handle. The handle's topk_idx (set via ncclEpUpdateHandle / ncclEpCreateHandle)
//                            is used by HT forward dispatch. For HT backward dispatch or LL mode,
//                            set topk_idx to NULL when calling ncclEpUpdateHandle.
//   inputs        - [IN]     Named input tensors (see ncclEpDispatchInputs_t).
//                            inputs->tokens is required; other fields are optional.
//   outputs       - [IN,OUT] Named preallocated output tensors (see ncclEpDispatchOutputs_t).
//                            outputs->tokens is required; other fields are optional.
//                            For HT (NCCL_EP_LAYOUT_FLAT): outputs->tokens is [N(r) x hidden] (2D),
//                                    where N(r) = num_ranks * max_dispatch_tokens_per_rank for static allocation,
//                                    or the actual received count when max_dispatch_tokens_per_rank is NCCL_EP_AUTO.
//                            For LL expert-major: outputs->tokens is [local_experts x num_recv_tokens x hidden] (3D).
//                            For LL rank-major: outputs->tokens is
//                                    [num_ranks x max_dispatch_tokens_per_rank x hidden] (3D) — the
//                                    per-rank, per-slot structure is explicit in the descriptor;
//                                    outputs->topk_weights and outputs->topk_idx must also be provided.
//   layout_info - [IN,OUT] Named local tensors (see ncclEpLayoutInfo_t). NULL = none.
//                            LL expert-major: layout_info->expert_counters receives per-expert token counts.
//                            LL rank-major: layout_info->src_rank_counters receives per-source-rank token counts.
//   config        - [IN]     Dispatch configuration (see ncclEpDispatchConfig_t). NULL = defaults.
//   stream        - [IN]     CUDA stream. If `ncclEpDispatch()` is called on a different stream than the stream used in
//                            `ncclEpCreateHandle()`,
//                            it is the responsibility of the user to synchronize between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpDispatch(
    ncclEpHandle_t handle,
    const ncclEpDispatchInputs_t* inputs,
    const ncclEpDispatchOutputs_t* outputs,
    const ncclEpLayoutInfo_t* layout_info,  // NULL = none
    const ncclEpDispatchConfig_t* config,   // NULL = defaults
    cudaStream_t stream
);

typedef struct {
    unsigned int size;         // = sizeof(this struct); first field, never moves
    unsigned int magic;        // = NCCL_EP_MAGIC; second field, never moves
    unsigned int send_only;    // if non-zero, only initiate transfers; requires ncclEpComplete() afterward
                               //   supported for LL mode only; output tensors must still be preallocated
    ncclEpPassDir_t pass_direction; // forward (default) or backward pass; HT-only.
                               //   FWD forbids inputs->topk_weights; BWD requires inputs->topk_weights
                               //   and outputs->topk_weights.
} ncclEpCombineConfig_t;

#define NCCL_EP_COMBINE_CONFIG_INIT ((ncclEpCombineConfig_t){ \
    .size  = (unsigned int)sizeof(ncclEpCombineConfig_t), \
    .magic = NCCL_EP_MAGIC })

// Perform EP combine
//   * Gathers outputs from experts and returns them to their source in original token order.
//   * This call is collective and must be invoked by all ranks in the group.
//
// Arguments:
//   handle           - [IN,OUT] EP handle that was used for `ncclEpDispatch()` operation
//   inputs           - [IN]     Named input tensors (see ncclEpCombineInputs_t).
//                               inputs->tokens is required; other fields are optional.
//                               For HT (NCCL_EP_LAYOUT_FLAT): inputs->tokens is [N(r) x hidden] (2D).
//                               For LL expert-major: inputs->tokens is [local_experts x num_recv_tokens x hidden] (3D).
//                               For LL rank-major: inputs->tokens is
//                                       [num_ranks x max_dispatch_tokens_per_rank x hidden] (3D) — the
//                                       per-rank, per-slot structure is explicit in the descriptor;
//                                       pre-reduced across local experts by the caller before this call.
//                               HT backward: inputs->topk_weights must also be provided.
//   outputs          - [IN,OUT] Named preallocated output tensors (see ncclEpCombineOutputs_t).
//                               outputs->tokens is required; 2D [num_tokens x hidden], restored to original order.
//                               outputs->topk_weights:
//                                 LL expert-major: per-token routing weights applied on the combine receive side.
//                                 HT backward: must also be provided; receives combined routing weights.
//   config           - [IN]     Combine configuration (see ncclEpCombineConfig_t). NULL = defaults.
//   stream           - [IN]     CUDA stream. If `ncclEpCombine()` is called on a different stream than the stream
//                               used in `ncclEpCreateHandle()`, it is the responsibility of the user to synchronize
//                               between streams to ensure correctness.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpCombine(
    ncclEpHandle_t handle,
    const ncclEpCombineInputs_t* inputs,
    const ncclEpCombineOutputs_t* outputs,
    const ncclEpCombineConfig_t* config,    // NULL = defaults
    cudaStream_t stream
);

// Reserved config struct for future options. Callers may pass NULL (defaults)
// or initialise via NCCL_EP_COMPLETE_CONFIG_INIT. The size/magic header fields
// follow the same ABI/initialisation rules as the other public structs and
// also give the type complete shape (cybind / pycparser require this), so the
// typedef stays struct-form rather than pointer-form — callers can spell
// pointer-to-const as `const ncclEpCompleteConfig_t*` naturally.
typedef struct ncclEpCompleteConfig {
    unsigned int size;  // = sizeof(this struct); first field, never moves
    unsigned int magic; // = NCCL_EP_MAGIC; second field, never moves
} ncclEpCompleteConfig_t;

#define NCCL_EP_COMPLETE_CONFIG_INIT ((ncclEpCompleteConfig_t){ \
    .size  = (unsigned int)sizeof(ncclEpCompleteConfig_t), \
    .magic = NCCL_EP_MAGIC })

// Continues a staged EP operation to completion.
//   * This should be called after a prior `ncclEpDispatch()` or `ncclEpCombine()` call with `send_only` flag set.
//
// Arguments:
//   handle     - [IN,OUT] EP handle used in the preceding staged operation
//   config     - [IN]     Completion configuration (see ncclEpCompleteConfig_t). NULL = defaults.
//   stream     - [IN]     CUDA stream
//
// Notes:
//   - If `ncclEpComplete()` is called on a different stream than the operation initiation call
//     (i.e., `ncclEpDispatch()` or `ncclEpCombine()`), it is the responsibility of the user to
//     synchronize between streams to ensure correctness.
//   - Only LL mode is supported.
//
// Returns:
//   ncclResult_t error code

ncclResult_t ncclEpComplete(
    ncclEpHandle_t handle,
    const ncclEpCompleteConfig_t* config,
    cudaStream_t stream
);


// Query the active-mask status of all ranks.
//   Copies the mask buffer to a user-provided device tensor.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group     - [IN]  EP group with masking enabled
//   mask_status  - [OUT] Device pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream       - [IN]  CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskQuery(
    ncclEpGroup_t ep_group,
    int* mask_status,
    cudaStream_t stream
);

// Set the mask for all ranks at once.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group   - [IN] EP group with masking enabled
//   mask       - [IN] Host pointer to int[nRanks]. 1 = active, 0 = masked (failed).
//   stream     - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskUpdate(
    ncclEpGroup_t ep_group,
    const int* mask,
    cudaStream_t stream
);

// Reset masks and RDMA buffers so previously masked ranks can re-join.
//   Collective: all surviving ranks must call simultaneously.
//   Resets RDMA buffers via a cross-rank barrier and sets all masks to active.
//   Does NOT reset the async error flag — call ncclEpErrorClear() separately.
//   Note: this API is for re-admitting a delayed rank within the same
//   communicator. Rank replacement requires a new communicator (e.g.,
//   ncclCommGrow) and a new EP group.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//   stream   - [IN] CUDA stream
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpMaskClean(
    ncclEpGroup_t ep_group,
    cudaStream_t stream
);

// Poll for asynchronous errors (e.g., rank timeout).
//   Lightweight host-side check — reads a pinned CPU flag, no GPU sync required.
//   The flag is set by the kernel when a timeout masks a rank; clear it
//   explicitly via ncclEpErrorClear().
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group  - [IN]  EP group with masking enabled
//   error_out - [OUT] 0 = no error, 1 = timeout occurred (one or more ranks masked)
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpGetAsyncError(
    ncclEpGroup_t ep_group,
    int* error_out
);

// Clear the async error flag.
//   Lightweight host-side reset — writes zero to the pinned CPU flag.
//   Use after detecting an error (via ncclEpGetAsyncError) to re-arm the flag
//   for detecting new failures. Should be called after ncclEpMaskClean (full
//   recovery) or standalone when surviving ranks continue in degraded mode.
//   Requires enable_mask=true in the group config.
//
// Arguments:
//   ep_group - [IN] EP group with masking enabled
//
// Returns: ncclResult_t error code

ncclResult_t ncclEpErrorClear(
    ncclEpGroup_t ep_group
);

#ifdef __cplusplus
}
#endif
