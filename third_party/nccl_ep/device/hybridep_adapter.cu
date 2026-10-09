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

#include "nccl_device.h"
#include "hybridep_adapter.cuh"
#include "hybridep_configs.cuh"
#include "hybrid_ep.cuh"
#include "include/common.hpp"
#include "jit/combine_jit.cuh"
#include "jit/dispatch_jit.cuh"
#include "jit/preprocess_jit.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
__global__ void convert_topk_to_routing_map_kernel(
    const int64_t* __restrict__ topk_idx,    // [num_tokens, num_topk]
    uint8_t* __restrict__ routing_bitmap,     // [max_tokens, num_experts_packed]
    int64_t* __restrict__ cached_topk_idx,    // [num_tokens, num_topk]; nullable
    int num_tokens,
    int max_tokens,                           // tail-zero bound (>= num_tokens)
    int num_topk,
    int num_experts_packed                    // = ceil(num_experts / 8)
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= max_tokens) return;

    // Each thread exclusively owns its row -- no atomics needed.
    // Zero the row before OR-ing in bits; the caller does not pre-zero.
    // Threads for tail rows [num_tokens, max_tokens) zero and exit, so the
    // downstream ncclAllGather over max_tokens rows ships clean tail bytes.
    uint8_t* row = routing_bitmap + token * num_experts_packed;
    for (int b = 0; b < num_experts_packed; b++) row[b] = 0;
    if (token >= num_tokens) return;
    const int64_t* in_row  = topk_idx + token * num_topk;
    int64_t*       out_row = cached_topk_idx ? cached_topk_idx + token * num_topk : nullptr;
    for (int k = 0; k < num_topk; k++) {
        int64_t expert = in_row[k];
        if (out_row) out_row[k] = expert;
        if (expert >= 0) {
            row[expert / 8] |= (1u << (expert % 8));
        }
    }
}

// ============================================================================
// Convert topk to bitmap routing map
// ============================================================================
void convert_topk_to_routing_map(
    const int64_t* topk_idx,
    uint8_t* routing_bitmap,
    int64_t* cached_topk_idx,
    int num_tokens,
    int max_tokens,
    int num_topk,
    int num_experts_packed,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (max_tokens + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, routing_bitmap, cached_topk_idx, num_tokens, max_tokens, num_topk, num_experts_packed);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob
// ============================================================================
__global__ void sparse_to_dense_prob_kernel(
    const int64_t* __restrict__ topk_idx,      // [num_tokens, topk]
    const float* __restrict__ topk_weights,    // [num_tokens, topk]
    float* __restrict__ dense_prob,            // [num_tokens, num_experts]
    int num_tokens,
    int num_topk,
    int num_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token = tid / num_topk;
    int k = tid % num_topk;

    if (token >= num_tokens) return;

    int64_t expert = topk_idx[token * num_topk + k];
    float weight = topk_weights[token * num_topk + k];

    // Scatter weight to the correct expert position
    if (expert >= 0 && expert < num_experts) {
        dense_prob[token * num_experts + expert] = weight;
    }
}

// ============================================================================
// Convert sparse to dense prob
// ============================================================================
void sparse_to_dense_prob(
    const int64_t* topk_idx,
    const float* topk_weights,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_topk;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    sparse_to_dense_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, topk_weights, dense_prob, num_tokens, num_topk, num_experts);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob for combine input
// ============================================================================
// Used for combine backward pass. Uses local_expert_routing_map to determine
// which experts each token is routed to, matching the order from dispatch output.
// Each thread handles one token.
__global__ void sparse_to_dense_prob_combine_kernel(
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const bool* __restrict__ local_expert_routing_map, // [num_tokens, experts_per_rank]
    float* __restrict__ dense_prob,                   // [num_tokens, experts_per_node]
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Scan local experts in order (matches dense_to_sparse_prob output order)
    int k_in = 0;
    for (int e = 0; e < experts_per_rank && k_in < num_topk; e++) {
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // This expert is active for this token - take next weight from sparse input
            float weight = topk_weights[token * num_topk + k_in];

            // Place at correct position in dense matrix
            // Local expert e on local_rank maps to: local_rank * experts_per_rank + e
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            dense_prob[dense_idx] = weight;

            k_in++;
        }
    }
}

// ============================================================================
// Convert sparse to dense prob for combine input
// ============================================================================
void sparse_to_dense_prob_combine(
    const float* topk_weights,
    const bool* local_expert_routing_map,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    sparse_to_dense_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_weights, local_expert_routing_map, dense_prob, num_tokens, num_topk,
        experts_per_rank, experts_per_node, local_rank);
}

// ============================================================================
// Kernel: Convert dense prob output to sparse format
// ============================================================================
// One thread per token. Output by layout:
//   FLAT/RM: recv_topk_weights[token, k_out] zero-filled tail; recv_topk_idx parallel.
//   EM:      recv_topk_weights[token] (single scalar; slot = (token, local_expert)); recv_topk_idx unused.
__global__ void dense_to_sparse_prob_kernel(
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_node]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // EM: [N]; FLAT/RM: [N, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]; nullptr under EM
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    bool expert_major
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_recv_tokens) return;

    if (expert_major) {
        // Each slot has at most one matching local expert (the one defining the slot).
        // Write the single scalar weight at recv_topk_weights[token]; default 0.
        float weight = 0.0f;
        for (int e = 0; e < experts_per_rank; e++) {
            if (local_expert_routing_map[token * experts_per_rank + e]) {
                int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
                weight = dense_prob[dense_idx];
                break;
            }
        }
        recv_topk_weights[token] = weight;
        return;
    }

    int k_out = 0;

    // Scan local experts (the ones this rank is responsible for)
    for (int e = 0; e < experts_per_rank && k_out < topk; e++) {
        // Check if this token is routed to expert e
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // Use local expert id for NCCL API compatibility (expects 0-based local indices)
            int64_t local_expert = static_cast<int64_t>(e);

            // Get weight from dense output (indexed by local expert within node)
            // dense_prob layout: [token, experts_per_node] where experts_per_node = experts_per_rank * ranks_per_node
            // Local rank's experts are at offset: local_rank * experts_per_rank
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            float weight = dense_prob[dense_idx];

            // Write outputs
            if (recv_topk_idx != nullptr) {
                recv_topk_idx[token * topk + k_out] = local_expert;
            }
            recv_topk_weights[token * topk + k_out] = weight;
            k_out++;
        }
    }

    // Zero-fill remaining topk slots if fewer than topk experts found
    for (; k_out < topk; k_out++) {
        if (recv_topk_idx != nullptr) {
            recv_topk_idx[token * topk + k_out] = -1;  // Invalid expert marker
        }
        recv_topk_weights[token * topk + k_out] = 0.0f;
    }
}

// O(top_k) lookup from cached_topk_idx; k-slot order preserves FWD input.
__global__ void dense_to_sparse_prob_combine_kernel(
    const float* __restrict__ dense_prob,         // [num_tokens, num_experts]
    const int64_t* __restrict__ cached_topk_idx,  // [num_tokens, topk]
    float* __restrict__ combined_topk_weights,    // [num_tokens, topk]
    int num_tokens,
    int topk,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    for (int k = 0; k < topk; k++) {
        int64_t e = cached_topk_idx[token * topk + k];
        float weight = (e >= 0 && e < num_experts)
            ? dense_prob[token * num_experts + e]
            : 0.0f;
        combined_topk_weights[token * topk + k] = weight;
    }
}

void dense_to_sparse_prob_combine(
    const float* dense_prob,
    const int64_t* cached_topk_idx,
    float* combined_topk_weights,
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, cached_topk_idx, combined_topk_weights,
        num_tokens, topk, num_experts);
}


// ============================================================================
// Dense to sparse prob
// ============================================================================
void dense_to_sparse_prob(
    const float* dense_prob,
    const bool* local_expert_routing_map,
    float* recv_topk_weights,
    int64_t* recv_topk_idx,
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    bool expert_major,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_recv_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, local_expert_routing_map, recv_topk_weights, recv_topk_idx,
        num_recv_tokens, topk, experts_per_rank, experts_per_node, local_rank,
        expert_major);
}

// ============================================================================
// Call metadata preprocessing
// ============================================================================
void call_metadata_preprocessing(
    const uint8_t* global_routing_map,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    void* token_rank_mask,
    int32_t* num_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    void* scan_tmp,
    int node_rank,
    int local_rank,
    int num_tokens_per_rank,
    int num_nodes,
    int num_ranks_per_node,
    int experts_per_rank,
    bool     expert_major,
    int64_t* internal_offsets,
    void*    padded_out_counts,
    void*    out_offsets,
    size_t alignment,
    int32_t* actual_counts_out,
    int s2d_inner_dim,
    void*    recv_total_counter,
    bool     out_is_int64,
    int      max_recv_tokens_per_rank,
    int      num_blocks,
    cudaStream_t stream
) {
    if (expert_major && per_expert_token_counts == nullptr) {
        EP_HOST_ASSERT(false && "EXPERT_MAJOR remap requires per_expert_token_counts != nullptr");
    }

    if (per_expert_token_counts != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(per_expert_token_counts, 0, experts_per_rank * sizeof(int32_t), stream));
    }

    constexpr int NUM_THREADS_PER_BLOCK = HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING;
    const int NUM_OF_BLOCKS = num_blocks;
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;

    const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(scan_tmp, 0, preprocessing_tmp_sz, stream));

    const size_t scan_smem_size =
        (2 * NUM_OF_WARPS_PER_BLOCK_SCAN * num_ranks_per_node * sizeof(int32_t)) +
        (num_ranks_per_node * sizeof(int32_t)) +
        (per_expert_token_counts != nullptr ? experts_per_rank * sizeof(int32_t) : 0);
    const size_t remap_smem_size = expert_major
        ? (static_cast<size_t>(experts_per_rank) * sizeof(int64_t) +
           static_cast<size_t>(NUM_OF_WARPS_PER_BLOCK_SCAN) * experts_per_rank * sizeof(int32_t))
        : 0;
    const int dynamic_smem_bytes = static_cast<int>(
        scan_smem_size > remap_smem_size ? scan_smem_size : remap_smem_size);

    ::hybrid_ep::scan_kernel_param_t sp;
    sp.input_routing_map = global_routing_map;
    sp.tmp = reinterpret_cast<::hybrid_ep::tmp_state_t*>(scan_tmp);
    sp.sparse_to_dense_map = sparse_to_dense_map;
    sp.rdma_to_attn_map = rdma_to_attn_map;
    sp.attn_to_rdma_map = attn_to_rdma_map;
    sp.token_rank_mask = token_rank_mask;
    sp.num_of_tokens_for_experts = num_tokens_for_experts;
    sp.local_expert_routing_map = local_expert_routing_map;
    sp.per_expert_token_counts = per_expert_token_counts;
    sp.node_rank = node_rank;
    sp.local_rank = local_rank;
    sp.num_of_tokens_per_rank = num_tokens_per_rank;
    sp.num_of_ranks_per_node = num_ranks_per_node;
    sp.experts_per_rank = experts_per_rank;
    sp.expert_major = expert_major;
    sp.remap_alignment = alignment;
    sp.remap_internal_offsets = internal_offsets;
    sp.remap_padded_out_counts = padded_out_counts;
    sp.remap_out_offsets = out_offsets;
    sp.remap_actual_counts_out = actual_counts_out;
    sp.s2d_inner_dim = s2d_inner_dim;
    sp.recv_total_counter = recv_total_counter;
    sp.out_is_int64 = out_is_int64;
    sp.max_recv_tokens_per_rank = max_recv_tokens_per_rank;

    jit::launch_scan(
        NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, num_nodes, num_ranks_per_node,
        per_expert_token_counts != nullptr, sp, dynamic_smem_bytes, stream);
}

size_t get_preprocessing_scan_tmp_size(int num_ranks_per_node) {
    return HYBRIDEP_NUM_BLOCKS_PREPROCESSING * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
}

size_t get_rank_mask_elem_size(int lsa_team_size) {
    return ((lsa_team_size + 63) / 64) * sizeof(uint64_t);
}

int get_device_max_dynamic_smem() {
    int device = 0;
    int max_smem = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return max_smem;
}

void check_dispatch_smem_limit(
    const ::hybrid_ep::dispatch_config_t& config,
    size_t smem_size) {
    const int max_smem = get_device_max_dynamic_smem();
    if (smem_size <= static_cast<size_t>(max_smem)) return;

    std::fprintf(
        stderr,
        "[nccl_ep] dispatch dynamic shared memory exceeds device limit: requested=%zu bytes, "
        "limit=%d bytes. Tune dispatch stages/pipelines; current stages=%d, pipelines=%d.\n",
        smem_size,
        max_smem,
        config.num_of_stages,
        config.num_pipelines);
    std::abort();
}

// ============================================================================
// Dispatch wrapper implementation
// ============================================================================

// Helper to populate the fixed-size dispatch parameter fields from DispatchParams.
template<typename TOKEN_DATA_TYPE>
::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>
build_dispatch_param_base(const DispatchParams& params) {
    ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = params.attn_input_scaling_factor;

    // Metadata and sync flags
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.pad_actual_counts = params.pad_actual_counts;
    kp.pad_expert_token_offsets = params.pad_expert_token_offsets;
    kp.pad_alignment = params.pad_alignment;
    kp.expected_rdma_flag_value = params.expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.intra_node_write_completion_flags;
    kp.dispatch_grid_barrier_counter = params.dispatch_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.sf_window = params.nccl_sf_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .attn_input_token_offset = params.mr_info.attn_input_token_offset,
               .attn_input_prob_offset = params.mr_info.attn_input_prob_offset,
               .attn_input_scaling_factor_offset = params.mr_info.attn_input_scaling_factor_offset,
               // Batched staging parameters (packed layout)
               .rdma_send_staging_offset = params.mr_info.rdma_send_staging_offset,
               .rdma_inter_node_group_packed_offset = params.mr_info.rdma_inter_node_group_packed_offset,
               .bytes_per_entry = params.mr_info.bytes_per_entry,
               .max_tokens_per_dest = params.mr_info.max_tokens_per_dest,
               // Streaming signal parameters
               .signals_tail_base = params.mr_info.signals_tail_base,
               .num_max_rdma_chunked_send_tokens = params.mr_info.num_max_rdma_chunked_send_tokens
            };

    return kp;
}

template<typename TOKEN_DATA_TYPE>
std::vector<uint8_t> build_dispatch_arg_buffer(
    const ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>& kp,
    const DispatchParams& params) {
    using ParamBase = ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.num_ranks_per_node * sizeof(TOKEN_DATA_TYPE*);
    const size_t sf_offset = prob_offset + params.num_ranks_per_node * sizeof(float*);
    const size_t total_size = sf_offset + params.num_ranks_per_node * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<TOKEN_DATA_TYPE**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    auto* sf_ptrs = reinterpret_cast<float**>(arg.data() + sf_offset);
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        token_ptrs[i] = reinterpret_cast<TOKEN_DATA_TYPE*>(params.expert_output_token_ptrs[i]);
        prob_ptrs[i] = params.expert_output_prob_ptrs ? params.expert_output_prob_ptrs[i] : nullptr;
        sf_ptrs[i] = params.expert_output_scaling_factor_ptrs ?
            params.expert_output_scaling_factor_ptrs[i] : nullptr;
    }

    return arg;
}

// Template dispatch launcher for forward/backward and sync modes
template<bool FORWARD_DISPATCH>
void dispatch_impl(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    int num_blocks,
    cudaStream_t stream
) {
    HYBRIDEP_SWITCH_DATATYPE(use_fp8, {
        // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
        // Check alignment at runtime now that experts_per_rank is dynamic
        const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
        assert((experts_per_node * sizeof(float)) % 16 == 0 &&
               "experts_per_node must be multiple of 4 for TMA alignment");
        // 16B cp.async.bulk alignment for the S2D map fetch; matters when s2d_inner_dim < 4.
        assert((static_cast<int64_t>(params.num_tokens_per_rank) * params.s2d_inner_dim) % 4 == 0 &&
               "Dispatch S2D cp.async.bulk: num_tokens_per_rank * s2d_inner_dim must be a "
               "multiple of 4 (flat layout with lsa_team_size <= 3 requires even num_tokens_per_rank)");

        auto kp = build_dispatch_param_base<TOKEN_DATA_TYPE>(params);
        constexpr bool kUseFp8 = std::is_same_v<TOKEN_DATA_TYPE, uint8_t>;

        // Compute dynamic SMEM size at host (was done inside hybrid_ep::dispatch).
        ::hybrid_ep::dispatch_config_t d_config;
        ::hybrid_ep::model_config_t   d_model;
        d_config.num_of_stages           = HYBRIDEP_DISPATCH_NUM_OF_STAGES;
        d_config.num_of_in_flight_s2g    = HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G;
        d_config.num_of_tokens_per_chunk = HT_OF_NUM_TOKENS_PER_CHUNK;
        d_config.num_of_blocks           = num_blocks;
        d_config.forward_dispatch        = FORWARD_DISPATCH;
        d_config.token_data_type         = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
        d_config.num_pipelines           = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.stages_per_pipeline     = HYBRIDEP_DISPATCH_NUM_OF_STAGES / HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.s2d_inner_dim           = kp.s2d_inner_dim;
        d_model.hidden_dim               = kp.hidden_dim;
        d_model.max_num_of_tokens_per_rank = MAX_SUPPORTED_TOKENS_PER_RANK;
        d_model.num_of_experts_per_rank  = kp.experts_per_rank;
        d_model.num_of_ranks_per_node    = kp.num_of_ranks_per_node;
        d_model.num_of_nodes             = num_nodes;

        const int smem_size = (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? ::hybrid_ep::calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_EXPERT_MAJOR>(d_config, d_model)
            : ::hybrid_ep::calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_FLAT>(d_config, d_model);
        check_dispatch_smem_limit(d_config, smem_size);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        const jit::dispatch_warp_layout_t dispatch_layout =
            jit::compute_dispatch_warp_layout(num_nodes, params.layout);
        const int dispatch_wt_total = num_blocks * (dispatch_layout.block_dim / 32);
        ::hybrid_ep::dispatch_warp_timing_entry_t* d_wt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wt, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t)));
        CUDA_CHECK(cudaMemsetAsync(d_wt, 0, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t), stream));
        kp.warp_timing = d_wt;
#endif

        std::vector<uint8_t> kernel_arg = build_dispatch_arg_buffer(kp, params);
        jit::launch_dispatch(
            HYBRIDEP_DISPATCH_NUM_OF_STAGES,
            HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G,
            HT_OF_NUM_TOKENS_PER_CHUNK,
            num_blocks,
            FORWARD_DISPATCH,
            num_nodes,
            params.num_ranks_per_node,
            params.layout,
            kUseFp8,
            kp.hidden_dim,
            kernel_arg.data(),
            kernel_arg.size(),
            smem_size,
            stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        jit::dispatch_dump_warp_timing(dispatch_layout, num_blocks, d_wt, stream);
        CUDA_CHECK(cudaFree(d_wt));
#endif
    });
}

void call_dispatch(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    bool forward_dispatch,
    int num_blocks,
    cudaStream_t stream
) {
    // Dispatch based on forward/backward and sync mode
    if (forward_dispatch) {
        dispatch_impl<true>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, stream);

    } else {
        dispatch_impl<false>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, stream);

    }
}

// ============================================================================
// Combine wrapper implementation
// ============================================================================

// Helper to populate the fixed-size combine parameter fields from CombineParams.
::hybrid_ep::combine_kernel_param_base_t
build_combine_param_base(const CombineParams& params) {
    ::hybrid_ep::combine_kernel_param_base_t kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User output buffers
    kp.attn_output_token = reinterpret_cast<uint16_t*>(params.attn_output_token);
    kp.attn_output_prob = params.attn_output_prob;

    // RDMA buffers (multi-node only)
    kp.rdma_intra_node_red_token = params.rdma_intra_node_red_token;
    kp.rdma_intra_node_red_prob = params.rdma_intra_node_red_prob;
    kp.rdma_inter_node_group_token = params.combine_rdma_inter_node_group_token;
    kp.rdma_inter_node_group_prob = params.combine_rdma_inter_node_group_prob;

    // Metadata
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;

    // Sync flags
    kp.expected_rdma_flag_value = params.combine_expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.combine_expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.combine_rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.combine_intra_node_write_completion_flags;
    kp.combine_grid_barrier_counter = params.combine_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;
    kp.num_real_tokens        = params.num_real_tokens;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    kp.combine_signal_offset = params.combine_signal_offset;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .rdma_intra_node_red_token_offset = params.mr_info.rdma_intra_node_red_token_offset,
               .combine_rdma_inter_node_group_token_offset = params.mr_info.combine_rdma_inter_node_group_token_offset,
               .rdma_intra_node_red_prob_offset = params.mr_info.rdma_intra_node_red_prob_offset,
               .combine_rdma_inter_node_group_prob_offset = params.mr_info.combine_rdma_inter_node_group_prob_offset
    };

    return kp;
}

std::vector<uint8_t> build_combine_arg_buffer(
    const ::hybrid_ep::combine_kernel_param_base_t& kp,
    const CombineParams& params) {
    using ParamBase = ::hybrid_ep::combine_kernel_param_base_t;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.num_ranks_per_node * sizeof(uint16_t*);
    const size_t total_size = prob_offset + params.num_ranks_per_node * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<uint16_t**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        token_ptrs[i] = params.expert_input_token_ptrs[i];
        prob_ptrs[i] = params.expert_input_prob_ptrs ? params.expert_input_prob_ptrs[i] : nullptr;
    }

    return arg;
}


// Template combine launcher for forward/backward
template<bool BACKWARD_COMBINE>
void combine_impl(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    int num_blocks,
    cudaStream_t stream
) {
    // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
    const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
    assert((experts_per_node * sizeof(float)) % 16 == 0 &&
           "experts_per_node must be multiple of 4 for TMA alignment");

    auto kp = build_combine_param_base(params);

    // Select config based on num_nodes (single-node: 12 stages/2 pipelines, multi-node: 5 stages/1 pipeline)
    const int num_stages_g2s = (num_nodes == 1)
        ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S
        : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S;
    const int num_stages_s2g = (num_nodes == 1)
        ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G
        : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G;

    ::hybrid_ep::model_config_t model;
    model.hidden_dim = kp.hidden_dim;
    model.max_num_of_tokens_per_rank = MAX_SUPPORTED_TOKENS_PER_RANK;
    model.num_of_experts_per_rank = kp.experts_per_rank;
    model.num_of_ranks_per_node = kp.num_of_ranks_per_node;
    model.num_of_nodes = num_nodes;
    const int smem_size = ::hybrid_ep::calculate_combine_smem_layout_size(
        num_stages_g2s,
        num_stages_s2g,
        HT_OF_NUM_TOKENS_PER_CHUNK,
        MAX_SUPPORTED_TOKENS_PER_RANK,
        num_nodes,
        BACKWARD_COMBINE,
        model);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    const jit::combine_warp_layout_t combine_layout = jit::compute_combine_warp_layout(num_nodes);
    const int combine_wt_total = num_blocks * (combine_layout.block_dim / 32);
    ::hybrid_ep::combine_warp_timing_entry_t* d_wt = nullptr;
    ::hybrid_ep::combine_block_timing_entry_t* d_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, num_blocks * sizeof(::hybrid_ep::combine_block_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_bt, 0, num_blocks * sizeof(::hybrid_ep::combine_block_timing_entry_t), stream));
    kp.warp_timing = d_wt;
    kp.block_timing = d_bt;
#endif

    std::vector<uint8_t> kernel_arg = build_combine_arg_buffer(kp, params);
    jit::launch_combine(
        num_stages_g2s,
        num_stages_s2g,
        HT_OF_NUM_TOKENS_PER_CHUNK,
        HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
        num_blocks,
        HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
        BACKWARD_COMBINE,
        num_nodes,
        params.num_ranks_per_node,
        params.layout,
        kp.hidden_dim,
        kernel_arg.data(),
        kernel_arg.size(),
        smem_size,
        stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    jit::combine_dump_warp_timing(combine_layout, num_blocks, d_wt, d_bt, stream);
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
}

void call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool backward_combine,
    int num_blocks,
    cudaStream_t stream
) {
    if (backward_combine) {
        combine_impl<true>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, num_blocks, stream);
    } else {
        combine_impl<false>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, num_blocks, stream);
    }
}

} // namespace hybridep
} // namespace nccl_ep
