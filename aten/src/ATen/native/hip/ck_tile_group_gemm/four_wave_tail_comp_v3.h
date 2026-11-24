#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>

namespace at::hip::detail::ck_tile_group_gemm {

    // BaseGemmPipelineAgBgCrCompV3 picks One/Even/Odd tails for eight-wave blocks, but CompV3's loop body only
    // implements the four-wave scheme (hot loop iff num_loop > 2, Even tail only for exactly two K tiles), so an
    // eight-wave CompV3 computes the wrong number of K tiles. GroupedGemmKernel calls the overload that makes that
    // selection; this makes the four-wave selection instead and passes it to CompV3's public overload that takes it.
    template <typename Problem>
    struct FourWaveTailCompV3 : ck_tile::GemmPipelineAgBgCrCompV3<Problem> {
        using CompV3 = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
        template <typename ADramWindow, typename BDramWindow>
        CK_TILE_DEVICE auto operator()(const ADramWindow& a_window, const BDramWindow& b_window, ck_tile::index_t num_loop, void* p_smem) const {
            const bool has_hot_loop = num_loop > CompV3::PrefetchStages;
            const auto tail_number = num_loop == CompV3::PrefetchStages ? ck_tile::TailNumber::Even : ck_tile::TailNumber::Odd;
            return CompV3::operator()(a_window, b_window, num_loop, has_hot_loop, tail_number, p_smem);
        }
    };

} // namespace at::hip::detail::ck_tile_group_gemm
