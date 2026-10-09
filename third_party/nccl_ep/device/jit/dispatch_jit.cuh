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

constexpr const char* kDispatchJitEntryName = "nccl_ep_jit_ht_dispatch_kernel";

inline const char* dispatch_bool_literal(bool value) { return value ? "true" : "false"; }

inline const char* dispatch_token_data_type_literal(bool use_fp8) {
    return use_fp8 ? "uint8_t" : "uint16_t";
}

struct dispatch_warp_layout_t {
    int inter_node_group_warps;
    int inter_node_group_start;
    int intra_node_g2s_group_warps;
    int intra_node_g2s_group_start;
    int intra_node_s2g_group_warps;
    int intra_node_s2g_group_start;
    int pad_group_warps;
    int pad_group_start;
    int num_pipelines;
    int block_dim;
};

inline dispatch_warp_layout_t compute_dispatch_warp_layout(int num_lsa_teams, ncclEpLayout_t layout) {
    const bool multinode_layout = (num_lsa_teams != 1);
    dispatch_warp_layout_t L{};
    L.num_pipelines = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
    L.inter_node_group_warps    = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    L.inter_node_group_start    = 0;
    L.intra_node_g2s_group_warps = L.num_pipelines;
    L.intra_node_g2s_group_start = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    L.intra_node_s2g_group_warps = L.num_pipelines;
    L.intra_node_s2g_group_start = multinode_layout
        ? (HYBRIDEP_DISPATCH_N2N_WARPS + L.num_pipelines)
        : L.num_pipelines;
    L.pad_group_warps = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? 1 : 0;
    L.pad_group_start = L.intra_node_s2g_group_start + L.intra_node_s2g_group_warps;
    L.block_dim = 32 * (
        L.inter_node_group_warps +
        L.intra_node_g2s_group_warps +
        L.intra_node_s2g_group_warps +
        L.pad_group_warps);
    return L;
}

inline std::string dispatch_jit_source(
    int inter_node_group_warps,
    int inter_node_group_start,
    int intra_node_g2s_group_warps,
    int intra_node_g2s_group_start,
    int intra_node_s2g_group_warps,
    int intra_node_s2g_group_start,
    int pad_group_warps,
    int pad_group_start,
    int num_of_stages,
    int num_of_in_flight_s2g,
    int num_of_tokens_per_chunk,
    int num_lsa_teams,
    int num_of_blocks,
    bool forward_dispatch,
    int num_pipelines,
    int lsa_team_size,
    ncclEpLayout_t layout,
    bool use_fp8,
    int hidden_dim) {
    const char* layout_literal =
        (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? "NCCL_EP_LAYOUT_EXPERT_MAJOR"
            : "NCCL_EP_LAYOUT_FLAT";
    const char* token_type_literal = dispatch_token_data_type_literal(use_fp8);
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << token_type_literal << ";\n"
        << "using INTER_NODE_GROUP     = hybrid_ep::warp_group<" << inter_node_group_warps    << ", " << inter_node_group_start    << ">;\n"
        << "using INTRA_NODE_G2S_GROUP = hybrid_ep::warp_group<" << intra_node_g2s_group_warps << ", " << intra_node_g2s_group_start << ">;\n"
        << "using INTRA_NODE_S2G_GROUP = hybrid_ep::warp_group<" << intra_node_s2g_group_warps << ", " << intra_node_s2g_group_start << ">;\n"
        << "using PAD_GROUP            = hybrid_ep::warp_group<" << pad_group_warps            << ", " << pad_group_start            << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size(), 1)\n"
        << "__global__ void " << kDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, "
        << lsa_team_size << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::dispatch_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      INTER_NODE_GROUP,\n"
        << "      INTRA_NODE_G2S_GROUP,\n"
        << "      INTRA_NODE_S2G_GROUP,\n"
        << "      PAD_GROUP,\n"
        << "      " << num_of_stages << ",\n"
        << "      " << num_of_in_flight_s2g << ",\n"
        << "      " << num_of_tokens_per_chunk << ",\n"
        << "      " << MAX_SUPPORTED_TOKENS_PER_RANK << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << dispatch_bool_literal(forward_dispatch) << ",\n"
        << "      " << num_pipelines << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << hidden_dim << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline void launch_dispatch(
    int num_of_stages,
    int num_of_in_flight_s2g,
    int num_of_tokens_per_chunk,
    int num_of_blocks,
    bool forward_dispatch,
    int num_lsa_teams,
    int lsa_team_size,
    ncclEpLayout_t layout,
    bool use_fp8,
    int hidden_dim,
    void* param,
    size_t param_size,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    const dispatch_warp_layout_t L = compute_dispatch_warp_layout(num_lsa_teams, layout);

    static const int fwd_variant_identity = 0;
    static const int bwd_variant_identity = 0;
    const int& variant_identity = forward_dispatch ? fwd_variant_identity : bwd_variant_identity;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "dispatch"
            << "_nodes" << num_lsa_teams
            << "_lsa" << lsa_team_size
            << "_hdim" << hidden_dim
            << "_stages" << num_of_stages
            << "_inflt" << num_of_in_flight_s2g
            << "_chunk" << num_of_tokens_per_chunk
            << "_blocks" << num_of_blocks
            << (forward_dispatch ? "_fwd" : "_bwd")
            << (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_fl")
            << (use_fp8 ? "_fp8" : "_bf16");
        return name.str();
    }();
    const std::string source = dispatch_jit_source(
        L.inter_node_group_warps,
        L.inter_node_group_start,
        L.intra_node_g2s_group_warps,
        L.intra_node_g2s_group_start,
        L.intra_node_s2g_group_warps,
        L.intra_node_s2g_group_start,
        L.pad_group_warps,
        L.pad_group_start,
        num_of_stages,
        num_of_in_flight_s2g,
        num_of_tokens_per_chunk,
        num_lsa_teams,
        num_of_blocks,
        forward_dispatch,
        L.num_pipelines,
        lsa_team_size,
        layout,
        use_fp8,
        hidden_dim);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_dispatch";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kDispatchJitEntryName;
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
            "[nccl_ep jit] fatal dispatch JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }
}

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
inline void dispatch_dump_warp_timing(
    const dispatch_warp_layout_t& L,
    int num_of_blocks,
    ::hybrid_ep::dispatch_warp_timing_entry_t* d_wt,
    cudaStream_t stream) {
    const int wt_warps_per_block = L.block_dim / 32;
    const int wt_total = num_of_blocks * wt_warps_per_block;
    char* pmix_rank_str = std::getenv("PMIX_RANK");
    int pmix_rank = pmix_rank_str ? std::atoi(pmix_rank_str) : -1;
    static int iter_count = 0;
    iter_count++;
    if (pmix_rank != 0 || iter_count != 40) return;

    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<::hybrid_ep::dispatch_warp_timing_entry_t> h_wt(wt_total);
    CUDA_CHECK(cudaMemcpy(h_wt.data(), d_wt, wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t), cudaMemcpyDeviceToHost));
    int _wt_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
    auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
    auto _wt_print_group = [&](const char* name, int warp_start, int warp_count) {
        if (warp_count == 0) return;
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        int n = 0;
        for (int b = 0; b < num_of_blocks; b++) {
            for (int w = warp_start; w < warp_start + warp_count; w++) {
                long long d = h_wt[b * wt_warps_per_block + w].end_clock -
                              h_wt[b * wt_warps_per_block + w].start_clock;
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
                if (e.start_clock < blk_start) blk_start = e.start_clock;
                if (e.end_clock > blk_end) blk_end = e.end_clock;
            }
            long long d = blk_end - blk_start;
            if (d < mn) mn = d;
            if (d > mx) mx = d;
            sum += d;
        }
        std::printf("[DISPATCH BLOCK SPAN TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                    num_of_blocks, _wt_us(mn), _wt_us(mx), _wt_us(sum / num_of_blocks));
    };
    std::printf("[DISPATCH WORK WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n",
                num_of_blocks, wt_warps_per_block, L.num_pipelines, _wt_clock_khz);
    _wt_print_group("INTER_N2N", L.inter_node_group_start, L.inter_node_group_warps);
    _wt_print_group("INTRA_G2S", L.intra_node_g2s_group_start, L.intra_node_g2s_group_warps);
    _wt_print_group("INTRA_S2G", L.intra_node_s2g_group_start, L.intra_node_s2g_group_warps);
    _wt_print_group("PAD",       L.pad_group_start,           L.pad_group_warps);
    _wt_print_block_span();
}
#endif

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
