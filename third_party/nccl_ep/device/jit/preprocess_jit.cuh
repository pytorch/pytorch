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
#include <sstream>
#include <string>

namespace nccl_ep {
namespace hybridep {
namespace jit {

constexpr const char* kScanJitEntryName = "nccl_ep_jit_ht_scan_kernel";

inline const char* scan_bool_literal(bool value) { return value ? "true" : "false"; }

inline std::string scan_jit_source(
    int num_threads_per_block,
    int num_of_blocks,
    int num_lsa_teams,
    int lsa_team_size,
    bool enable_per_expert_counts) {
    constexpr int rank_mask_word_bits = CHAR_BIT * static_cast<int>(sizeof(uint64_t));
    const int rank_mask_words = (lsa_team_size + rank_mask_word_bits - 1) / rank_mask_word_bits;
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << num_threads_per_block << ", 1)\n"
        << "__global__ void " << kScanJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::scan_kernel_param_t p) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::scan_impl<\n"
        << "      " << num_threads_per_block << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << scan_bool_literal(enable_per_expert_counts) << ">(\n"
        << "      p.input_routing_map, p.tmp, p.sparse_to_dense_map, p.rdma_to_attn_map, p.attn_to_rdma_map,\n"
        << "      reinterpret_cast<hybrid_ep::rank_mask_t<" << rank_mask_words << ">*>(p.token_rank_mask),\n"
        << "      p.num_of_tokens_for_experts, p.local_expert_routing_map, p.per_expert_token_counts,\n"
        << "      p.node_rank, p.local_rank, p.num_of_tokens_per_rank, p.num_of_ranks_per_node, p.experts_per_rank,\n"
        << "      p.expert_major,\n"
        << "      p.remap_alignment, p.remap_internal_offsets, p.remap_padded_out_counts, p.remap_out_offsets,\n"
        << "      p.remap_actual_counts_out, p.s2d_inner_dim, p.recv_total_counter, p.out_is_int64,\n"
        << "      p.max_recv_tokens_per_rank, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline void launch_scan(
    int num_threads_per_block,
    int num_of_blocks,
    int num_lsa_teams,
    int lsa_team_size,
    bool enable_per_expert_counts,
    ::hybrid_ep::scan_kernel_param_t& param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "scan"
            << "_nodes" << num_lsa_teams
            << "_lsa" << lsa_team_size
            << "_threads" << num_threads_per_block
            << "_blocks" << num_of_blocks
            << (enable_per_expert_counts ? "_pec" : "_nopec");
        return name.str();
    }();
    const std::string source = scan_jit_source(
        num_threads_per_block, num_of_blocks, num_lsa_teams, lsa_team_size, enable_per_expert_counts);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_scan";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kScanJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = 0;
    variant.num_blocks = num_of_blocks;
    variant.block_dim = num_threads_per_block;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal scan JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
