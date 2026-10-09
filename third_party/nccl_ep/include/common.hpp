/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// ============================================================================
// Configuration constants
// ============================================================================

#define NUM_MAX_RDMA_PEERS 20
#define NUM_WORKSPACE_BYTES (32 * 1024 * 1024)
#define NUM_MAX_LOCAL_EXPERTS 1024
#define NUM_BUFFER_ALIGNMENT_BYTES 128
#define MAX_HIDDEN_DIM 16384
#define MAX_NUM_TOPK 32
#define MAX_NCCL_GIN_CTX_PER_COMM 4
#define NUM_WAIT_NANOSECONDS 500
#define MAX_SUPPORTED_TOKENS_PER_RANK 8192  // Must match kernel template in hybridep_adapter.cu
#define FINISHED_SUM_TAG (MAX_SUPPORTED_TOKENS_PER_RANK * 2)
#define HT_OF_NUM_TOKENS_PER_CHUNK 64

// Timeout for GPU-side wait loops. When exceeded, the peer is masked (if active-mask
// is enabled) or the kernel traps. Setting this too low risks false positives: a rank
// that is merely slow may be marked as failed. Asymmetric timeouts across ranks can
// produce inconsistent masks (rank A masks rank B, but rank B does not mask rank A).
// Mask consistency is a framework-level concern -- the application should query the
// mask after detecting an error and reconcile as needed (e.g., via EPLB rebalance).
#ifndef ENABLE_FAST_DEBUG
#define NUM_CPU_TIMEOUT_SECS 100
#define NUM_TIMEOUT_CYCLES 200000000000ull // 200G cycles ~= 100s
#else
#define NUM_CPU_TIMEOUT_SECS 10
#define NUM_TIMEOUT_CYCLES 20000000000ull // 20G cycles ~= 10s
#endif

#define LOW_LATENCY_SEND_PHASE 1
#define LOW_LATENCY_RECV_PHASE 2

// NCCL GIN Configuration

// Make CLion CUDA indexing work
#ifdef __CLION_IDE__
#define __CUDA_ARCH__ 900 // NOLINT(*-reserved-identifier)
#define __CUDACC_RDC__ // NOLINT(*-reserved-identifier)
#endif

// Remove Torch restrictions
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#endif

// ============================================================================
// Standard includes
// ============================================================================
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "nccl.h"
#include "nccl_device.h"
#include "device/macros.cuh"
#include "ep_enums.h"

namespace nccl_ep {

// Internode low-latency kernels
namespace internode_ll {


// Helper function for alignment (host/device compatible)
template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
    return ((a + b - 1) / b) * b;
}

// Per-hop routing entry in the dispatch message header.
// Rank-major carries the topk weight on the wire so the receiver can write
// outRecvTopkWeights without an extra round-trip. Expert-major omits it,
// keeping the per-entry overhead at 2 bytes regardless of num_topk.
template <ncclEpLayout_t kLayout>
struct DispatchRouter;

template <>
struct DispatchRouter<NCCL_EP_LAYOUT_RANK_MAJOR> {
    float    topk_weight;  // written to outRecvTopkWeights on the receive side
    uint16_t expert_id;
    // sizeof = 8: float(4) + uint16_t(2) + 2 bytes implicit pad
};

template <>
struct DispatchRouter<NCCL_EP_LAYOUT_EXPERT_MAJOR> {
    uint16_t expert_id;
    // sizeof = 2
};

static_assert(sizeof(DispatchRouter<NCCL_EP_LAYOUT_RANK_MAJOR>)   == 8, "unexpected rank-major router size");
static_assert(sizeof(DispatchRouter<NCCL_EP_LAYOUT_EXPERT_MAJOR>) == 2, "unexpected expert-major router size");

// Dispatch message header: token identity + per-topk routing entries.
// alignas(16): header objects are 16-byte aligned in the RDMA buffer.
template <ncclEpLayout_t kLayout>
struct alignas(16) DispatchHdr {
    int token_id;
    DispatchRouter<kLayout> rtr[];  // Flexible array member (Note: this is not a C++ standard, but a CUDA extension)
};

static_assert(offsetof(DispatchHdr<NCCL_EP_LAYOUT_RANK_MAJOR>,   rtr) == 4, "unexpected rank-major rtr offset");
static_assert(offsetof(DispatchHdr<NCCL_EP_LAYOUT_EXPERT_MAJOR>, rtr) == 4, "unexpected expert-major rtr offset");

// Dispatch header wire size: token_id prefix through last rtr entry, rounded up
// to int4 (16-byte) boundary for vectorized RDMA access.
template <ncclEpLayout_t kLayout>
__host__ __device__ __forceinline__
size_t get_dispatch_hdr_sz(int num_topk) {
    const size_t base_sz = offsetof(DispatchHdr<kLayout>, rtr) +
                           static_cast<size_t>(num_topk) * sizeof(DispatchRouter<kLayout>);
    return align<size_t>(base_sz, sizeof(int4));
}

// Runtime overload: selects the correct template specialisation based on layout.
__host__ __forceinline__
size_t get_dispatch_hdr_sz(int num_topk, ncclEpLayout_t layout) {
    return layout == NCCL_EP_LAYOUT_RANK_MAJOR
        ? get_dispatch_hdr_sz<NCCL_EP_LAYOUT_RANK_MAJOR>(num_topk)
        : get_dispatch_hdr_sz<NCCL_EP_LAYOUT_EXPERT_MAJOR>(num_topk);
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              int* rankMask,
                              int* syncBuffer, ncclWindow_t* syncWindow,
                              ncclDevComm* devComms,
                              unsigned barrierSignalBase,
                              uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES,
                              cudaStream_t stream = 0);

void dispatch(const void* inData,
              const int64_t* inTopkIdx,
              const float* inTopkWeights,    // rank-major: written into dispatch message header
              void* outDataBuf,
              void* outScalesBuf,
              int* outSrcInfo,
              int* outRecvRankCounter,          // rank-major: RECV_RANK_COUNTER_DEVICE [nRanks]; nullptr for expert-major
              int64_t* outLayout,
              int* outCnt,
              float* outRecvTopkWeights,     // rank-major: received topk weights; nullptr for expert-major
              int32_t* outRecvTopkIdx,       // rank-major: received topk indices;  nullptr for expert-major
              void* sendBuf,
              void* recvBuf,
              int* recvCntBuf,
              size_t sendOff,
              size_t recvOff,
              size_t recvCntOff,
              int* nextRecvCntBuf,
              int nextRecvCntBufSize,
              int* recvStats,
              int64_t* waitStats,
              int numTokens,
              int hidden,
              int maxTokensPerRank,
              int numTopk,
              int numExperts,
              int currRank,
              int numRanks,
              bool use_fp8,
              bool roundScale,
              bool use_ue8m0,
              ncclEpLayout_t layout,
              int phases,
              int numComms,
              ncclDevComm* devComms,
              const ncclWindow_t* windows,
              unsigned signalsBase,
              void* workspace,
              int num_device_sms,
              int* rankMask = nullptr,
              int* asyncErrorFlag = nullptr,
              uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES,
              bool nvlinkOnly = false,
              cudaStream_t stream = 0);

void combine(const void* inData,
             const int* srcInfo,
             const int64_t* layoutRange,
             const int64_t* inTopkIdx,
             const float* topkWeights,
             void* outData,
             void* sendBuf,
             void* recvBuf,
             int* recvFlagBuf,
             size_t sendOff,
             size_t recvOff,
             size_t recvFlagOff,
             int* nextRecvCntBuf,
             int nextRecvCntBufSize,
             int64_t* waitStats,
             int numCombinedTokens,
             int hidden,
             int maxTokensPerRank,
             int numTopk,
             int numExperts,
             int currRank,
             int numRanks,
             bool useLogFmt,
             ncclEpLayout_t layout,
             int phases,
             bool zeroCopy,
             int numComms,
             ncclDevComm* devComms,
             const ncclWindow_t* windows,
             unsigned signalsBase,
             void* workspace,
             int num_device_sms,
             int* rankMask = nullptr,
             int* asyncErrorFlag = nullptr,
             uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES,
             cudaStream_t stream = 0);

} // namespace internode_ll

template <typename dtype_t>
__host__ __device__ constexpr dtype_t ceil_div(dtype_t a, dtype_t b) {
    return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t align(dtype_t a, dtype_t b) {
    return ceil_div<dtype_t>(a, b) * b;
}

// Buffer-relative offsets into rdma_buffer. Pointers are resolved at use time
// against the group's current rdma_buffer base, so the buffer can be
// reallocated (e.g., grown for a larger layout) without invalidating any
// cached layout state on live handles.
struct LowLatencyBuffer {
    int num_clean_int = 0;

    size_t dispatch_rdma_send_buffer_offset = 0;
    size_t dispatch_rdma_recv_data_buffer_offset = 0;
    size_t dispatch_rdma_recv_count_buffer_offset = 0;

    size_t combine_rdma_send_buffer_offset = 0;
    size_t combine_rdma_recv_data_buffer_offset = 0;
    size_t combine_rdma_recv_flag_buffer_offset = 0;

    size_t combine_rdma_send_buffer_data_start_offset = 0;
    size_t num_bytes_per_combine_msg = 0;

    // (count-buffer offset, count) for clean_low_latency_buffer. Dispatch
    // recv-count and combine recv-flag share the same signaling slot.
    std::pair<size_t, int> clean_meta_offset() const {
        EP_HOST_ASSERT(dispatch_rdma_recv_count_buffer_offset == combine_rdma_recv_flag_buffer_offset);
        return {dispatch_rdma_recv_count_buffer_offset, num_clean_int};
    }
};

struct LowLatencyLayout {
    size_t total_bytes = 0;
    LowLatencyBuffer buffers[2];

    template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
    static out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
        return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
    }

    LowLatencyLayout(int num_max_dispatch_tokens_per_rank, size_t max_token_bytes, int num_ranks, int num_experts, int num_topk, ncclEpLayout_t layout) {
        // Dispatch and combine layout:
        //  - 2 symmetric odd/even send buffer
        //  - 2 symmetric odd/even receive buffers
        //  - 2 symmetric odd/even signaling buffers

        // Per-slot sizes for buffer allocation (datatype-agnostic at the API;
        // max_token_bytes upper-bounds the per-token payload). The library's FP8
        // path quantizes from bf16 internally, so its per-token footprint is
        // bounded by max_token_bytes (which the caller sizes for the bf16 worst case).
        // Combine reserves additional per-128-bf16-element scale-factor space
        // (min/max as bf162) — an internal kernel contract of the FP8 quantizer.
        //
        // Per-call enforcement: ncclEpDispatch host-side asserts that the actual
        // input tokens fit max_token_bytes (unquantized path), and that the
        // FP8-quantized bytes fit max_token_bytes when the FP8 quantizer is engaged.
        // So the formulas below safely upper-bound any per-call kernel stride.
        const size_t num_scales = max_token_bytes / sizeof(nv_bfloat16) / 128;  // bf16 hidden / scale-tile size
        size_t scale_metadata_bytes = num_scales * sizeof(nv_bfloat162);

        size_t disp_hdr_sz = internode_ll::get_dispatch_hdr_sz(num_topk, layout);
        size_t num_bytes_per_dispatch_msg = disp_hdr_sz + max_token_bytes;
        size_t num_bytes_per_combine_msg = scale_metadata_bytes + max_token_bytes;

        // Per-op send and recv sizes.
        // Combine send buffer holds one slot per outbound message produced on
        // the sender side: expert-major fans tokens out per expert (num_experts
        // slots), while rank-major fans them out per destination rank
        // (num_ranks slots). Sizing per layout avoids over-allocating in the
        // rank-major case (typically num_ranks < num_experts).
        size_t combine_send_slots = (layout == NCCL_EP_LAYOUT_RANK_MAJOR)
            ? static_cast<size_t>(num_ranks)
            : static_cast<size_t>(num_experts);
        size_t dispatch_send_buffer_bytes = num_max_dispatch_tokens_per_rank * num_bytes_per_dispatch_msg;
        size_t combine_send_buffer_bytes  = combine_send_slots * num_max_dispatch_tokens_per_rank * num_bytes_per_combine_msg;

        // Symmetric receive buffers: the RDMA wire is always rank-major regardless of user-facing layout.
        size_t dispatch_recv_data_buffer_bytes =
            static_cast<size_t>(num_ranks) * static_cast<size_t>(num_max_dispatch_tokens_per_rank) * num_bytes_per_dispatch_msg;
        size_t combine_recv_buffer_bytes =
            static_cast<size_t>(num_max_dispatch_tokens_per_rank) * num_bytes_per_combine_msg * static_cast<size_t>(num_topk);

        // All four buffers feed vectorized int4 loads/stores; their starts (and
        // therefore their sizes, since they share slots) must be 16-byte aligned.
        EP_HOST_ASSERT(dispatch_send_buffer_bytes      % sizeof(int4) == 0);
        EP_HOST_ASSERT(dispatch_recv_data_buffer_bytes % sizeof(int4) == 0);
        EP_HOST_ASSERT(combine_send_buffer_bytes       % sizeof(int4) == 0);
        EP_HOST_ASSERT(combine_recv_buffer_bytes       % sizeof(int4) == 0);

        // Dispatch and combine never run concurrently and don't carry state in
        // their recv buffers across the boundary (each kernel copies received
        // payloads out to user tensors before exit). So one shared slot can
        // hold either op's (send + recv) pair, sized to the worst case.
        // Inside a slot:
        //   - send_buffer always starts at slot_base.
        //   - dispatch_recv starts at slot_base + dispatch_send_buffer_bytes.
        //   - combine_recv  starts at slot_base + combine_send_buffer_bytes.
        // Two slots for double-buffering.
        size_t dispatch_combo_bytes = dispatch_send_buffer_bytes + dispatch_recv_data_buffer_bytes;
        size_t combine_combo_bytes  = combine_send_buffer_bytes  + combine_recv_buffer_bytes;
        size_t slot_bytes           = std::max(dispatch_combo_bytes, combine_combo_bytes);
        total_bytes += slot_bytes * 2;

        // Symmetric signaling buffers
        size_t dispatch_recv_count_buffer_bytes = num_experts * sizeof(int);
        size_t combine_recv_flag_buffer_bytes = dispatch_recv_count_buffer_bytes;
        size_t signaling_buffer_bytes = std::max(dispatch_recv_count_buffer_bytes, combine_recv_flag_buffer_bytes);
        size_t signaling_buffer_bytes_aligned = align<size_t>(signaling_buffer_bytes, 128);
        total_bytes += signaling_buffer_bytes_aligned * 2;

        // Assign offsets into rdma_buffer. Pointers are resolved at use time
        // by adding these offsets to the group's current rdma_buffer base.

        for (int i = 0; i < 2; ++ i) {
            const size_t slot_base = signaling_buffer_bytes_aligned * 2 + slot_bytes * i;
            const size_t count_off = signaling_buffer_bytes_aligned * i;
            buffers[i] = LowLatencyBuffer{
                .num_clean_int                              = static_cast<int>(signaling_buffer_bytes / sizeof(int)),
                .dispatch_rdma_send_buffer_offset           = slot_base,
                .dispatch_rdma_recv_data_buffer_offset      = slot_base + dispatch_send_buffer_bytes,
                .dispatch_rdma_recv_count_buffer_offset     = count_off,
                .combine_rdma_send_buffer_offset            = slot_base,
                .combine_rdma_recv_data_buffer_offset       = slot_base + combine_send_buffer_bytes,
                .combine_rdma_recv_flag_buffer_offset       = count_off,
                .combine_rdma_send_buffer_data_start_offset = slot_base,
                .num_bytes_per_combine_msg                  = num_bytes_per_combine_msg,
            };
        }
    }
};

// Size for a specific LL layout, rounded up to the buffer alignment.
inline unsigned long int get_low_latency_rdma_size_hint_for_layout(
        int num_max_dispatch_tokens_per_rank, size_t max_token_bytes,
        int num_ranks, int num_experts, ncclEpLayout_t layout) {
    auto num_bytes = LowLatencyLayout(num_max_dispatch_tokens_per_rank,
                                      max_token_bytes, num_ranks, num_experts,
                                      MAX_NUM_TOPK, layout).total_bytes;
    return ((num_bytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

// Worst case across all LL layouts (upper bound on what may be needed once the
// per-handle layout is known). Used as the upper limit for the configured
// rdma_buffer_size; the actual allocation may start smaller and grow.
inline unsigned long int get_low_latency_rdma_size_hint(int num_max_dispatch_tokens_per_rank, size_t max_token_bytes, int num_ranks, int num_experts) {
    auto bytes_for = [&](ncclEpLayout_t layout) {
        return get_low_latency_rdma_size_hint_for_layout(
            num_max_dispatch_tokens_per_rank, max_token_bytes,
            num_ranks, num_experts, layout);
    };
    return std::max(bytes_for(NCCL_EP_LAYOUT_EXPERT_MAJOR),
                    bytes_for(NCCL_EP_LAYOUT_RANK_MAJOR));
}

} // namespace nccl_ep

