/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/hybrid_ep.cuh"
#include "device/jit/jit_runtime.hpp"

#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace nccl_ep {
namespace hybridep {
namespace jit {

constexpr const char* kCombineJitEntryName = "nccl_ep_jit_ht_combine_kernel";

inline const char* bool_literal(bool value) {
    return value ? "true" : "false";
}

struct combine_warp_layout_t {
    int intra_node_red_group_warps;
    int intra_node_red_group_start;
    int inter_node_red_group_warps;
    int inter_node_red_group_start;
    int intra_node_g2s_group_warps;
    int intra_node_g2s_group_start;
    int inter_node_g2s_group_warps;
    int inter_node_g2s_group_start;
    int inter_node_rdma_group_warps;
    int inter_node_rdma_group_start;
    int num_of_data_pipeline_per_block;
    int block_dim;
};

inline combine_warp_layout_t compute_combine_warp_layout(int num_lsa_teams) {
    const bool multinode_layout = (num_lsa_teams != 1);
    combine_warp_layout_t L{};
    L.intra_node_red_group_warps = multinode_layout ? 4 : 0;
    L.intra_node_red_group_start = 0;
    L.inter_node_red_group_warps = 4;
    L.inter_node_red_group_start = multinode_layout ? 4 : 0;
    L.intra_node_g2s_group_warps = multinode_layout ? 1 : 0;
    L.intra_node_g2s_group_start = multinode_layout ? 8 : 4;
    L.inter_node_g2s_group_warps = multinode_layout ? 1 : 2;
    L.inter_node_g2s_group_start = multinode_layout ? 9 : 4;
    L.inter_node_rdma_group_warps = multinode_layout ? 1 : 0;
    L.inter_node_rdma_group_start = multinode_layout ? 10 : 6;
    L.num_of_data_pipeline_per_block = multinode_layout ? 1 : 2;
    L.block_dim = 32 * (
        L.intra_node_red_group_warps +
        L.inter_node_red_group_warps +
        L.intra_node_g2s_group_warps +
        L.inter_node_g2s_group_warps +
        L.inter_node_rdma_group_warps);
    return L;
}

inline std::string combine_jit_source(
    const combine_warp_layout_t& L,
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_group,
    int num_of_tokens_per_chunk,
    int num_lsa_teams,
    int num_of_blocks,
    int num_of_additional_in_flight_s2g,
    bool backward_combine,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim) {
    const char* layout_literal =
        (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? "NCCL_EP_LAYOUT_EXPERT_MAJOR"
            : "NCCL_EP_LAYOUT_FLAT";
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "using INTRA_NODE_RED_GROUP = hybrid_ep::warp_group<" << L.intra_node_red_group_warps << ", " << L.intra_node_red_group_start << ">;\n"
        << "using INTER_NODE_RED_GROUP = hybrid_ep::warp_group<" << L.inter_node_red_group_warps << ", " << L.inter_node_red_group_start << ">;\n"
        << "using INTRA_NODE_G2S_GROUP = hybrid_ep::warp_group<" << L.intra_node_g2s_group_warps << ", " << L.intra_node_g2s_group_start << ">;\n"
        << "using INTER_NODE_G2S_GROUP = hybrid_ep::warp_group<" << L.inter_node_g2s_group_warps << ", " << L.inter_node_g2s_group_start << ">;\n"
        << "using INTER_NODE_RDMA_GROUP = hybrid_ep::warp_group<" << L.inter_node_rdma_group_warps << ", " << L.inter_node_rdma_group_start << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size(), 1)\n"
        << "__global__ void " << kCombineJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::combine_kernel_param_t<"
        << lsa_team_size << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::combine_kernel_impl<\n"
        << "      INTRA_NODE_RED_GROUP,\n"
        << "      INTER_NODE_RED_GROUP,\n"
        << "      INTRA_NODE_G2S_GROUP,\n"
        << "      INTER_NODE_G2S_GROUP,\n"
        << "      INTER_NODE_RDMA_GROUP,\n"
        << "      " << L.num_of_data_pipeline_per_block << ",\n"
        << "      " << num_of_stages_g2s << ",\n"
        << "      " << num_of_stages_s2g << ",\n"
        << "      " << num_of_tokens_per_group << ",\n"
        << "      " << num_of_tokens_per_chunk << ",\n"
        << "      " << MAX_SUPPORTED_TOKENS_PER_RANK << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << num_of_additional_in_flight_s2g << ",\n"
        << "      " << bool_literal(backward_combine) << ",\n"
        << "      " << hidden_dim << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << layout_literal << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline void launch_combine(
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_chunk,
    int num_of_tokens_per_group,
    int num_of_blocks,
    int num_of_additional_in_flight_s2g,
    bool backward_combine,
    int num_lsa_teams,
    int lsa_team_size,
    ncclEpLayout_t layout,
    int hidden_dim,
    void* param, // ptr to the packed kernel arguments buffer
    size_t param_size, // size of packed kernel arguments buffer
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    const combine_warp_layout_t L = compute_combine_warp_layout(num_lsa_teams);

    static const int fwd_variant_identity = 0;
    static const int bwd_variant_identity = 0;
    const int& variant_identity = backward_combine ? bwd_variant_identity : fwd_variant_identity;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "combine"
            << "_nodes" << num_lsa_teams
            << "_lsa" << lsa_team_size
            << "_hdim" << hidden_dim
            << "_g2s" << num_of_stages_g2s
            << "_s2g" << num_of_stages_s2g
            << "_chunk" << num_of_tokens_per_chunk
            << "_group" << num_of_tokens_per_group
            << "_blocks" << num_of_blocks
            << "_extra" << num_of_additional_in_flight_s2g
            << (backward_combine ? "_bwd" : "_fwd")
            << (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_fl");
        return name.str();
    }();
    const std::string source = combine_jit_source(
        L,
        num_of_stages_g2s,
        num_of_stages_s2g,
        num_of_tokens_per_group,
        num_of_tokens_per_chunk,
        num_lsa_teams,
        num_of_blocks,
        num_of_additional_in_flight_s2g,
        backward_combine,
        lsa_team_size,
        layout,
        hidden_dim);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_combine";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kCombineJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = num_of_blocks;
    variant.block_dim = L.block_dim;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, param, param_size, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal combine JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }
}

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
inline void combine_dump_warp_timing(
    const combine_warp_layout_t& L,
    int num_of_blocks,
    ::hybrid_ep::combine_warp_timing_entry_t* d_wt,
    ::hybrid_ep::combine_block_timing_entry_t* d_bt,
    cudaStream_t stream) {
    const int wt_warps_per_block = L.block_dim / 32;
    const int wt_total = num_of_blocks * wt_warps_per_block;
    char* pmix_rank_str = std::getenv("PMIX_RANK");
    int pmix_rank = pmix_rank_str ? std::atoi(pmix_rank_str) : -1;
    static int iter_count = 0;
    iter_count++;
    if (pmix_rank != 0 || iter_count != 40) return;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<::hybrid_ep::combine_warp_timing_entry_t> h_wt(wt_total);
    std::vector<::hybrid_ep::combine_block_timing_entry_t> h_bt(num_of_blocks);
    CUDA_CHECK(cudaMemcpy(h_wt.data(), d_wt, wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bt.data(), d_bt, num_of_blocks * sizeof(::hybrid_ep::combine_block_timing_entry_t), cudaMemcpyDeviceToHost));
    int _wt_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
    auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
    auto _wt_print_head_sync = [&]() {
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            long long d = h_bt[b].head_sync_end_clock - h_bt[b].head_sync_start_clock;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[COMBINE HEAD SYNC TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    auto _wt_print_work_group = [&](const char* name, int warp_start, int warp_count) {
        if (warp_count == 0) return;
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        int n = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            for (int w = warp_start; w < warp_start + warp_count; w++) {
                long long d = h_wt[b * wt_warps_per_block + w].work_end_clock -
                              h_wt[b * wt_warps_per_block + w].work_start_clock;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                sum += d;
                n++;
            }
        }
        std::printf("  %-9s (%d warp%s x %d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    name, warp_count, warp_count > 1 ? "s" : " ", num_of_blocks,
                    _wt_us(mn), _wt_us(mx), _wt_us(sum / n));
    };
    auto _wt_print_block_span = [&]() {
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            long long blk_start = LLONG_MAX;
            long long blk_end = 0;
            for (int w = 0; w < wt_warps_per_block; w++) {
                const auto& e = h_wt[b * wt_warps_per_block + w];
                if (e.work_start_clock < blk_start) blk_start = e.work_start_clock;
                if (e.work_end_clock > blk_end) blk_end = e.work_end_clock;
            }
            long long d = blk_end - blk_start;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[COMBINE BLOCK SPAN TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    _wt_print_head_sync();
    std::printf("[COMBINE WORK WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n",
                num_of_blocks, wt_warps_per_block, L.num_of_data_pipeline_per_block, _wt_clock_khz);
    _wt_print_work_group("INTRA_RED", L.intra_node_red_group_start, L.intra_node_red_group_warps);
    _wt_print_work_group("INTER_RED", L.inter_node_red_group_start, L.inter_node_red_group_warps);
    _wt_print_work_group("INTRA_G2S", L.intra_node_g2s_group_start, L.intra_node_g2s_group_warps);
    _wt_print_work_group("INTER_G2S", L.inter_node_g2s_group_start, L.inter_node_g2s_group_warps);
    _wt_print_work_group("INTER_N2N", L.inter_node_rdma_group_start, L.inter_node_rdma_group_warps);
    _wt_print_block_span();
}
#endif

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
