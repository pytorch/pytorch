#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>

namespace at::hip::detail::ck_tile_group_gemm {

	// BaseGemmPipelineAgBgCrCompV3 picks One/Even/Odd tails for eight-wave blocks, but CompV3's loop body only
	// implements the four-wave scheme (hot loop iff num_loop > 2, Even tail only for exactly two K tiles), so an
	// eight-wave CompV3 runs one K tile too many or too few. This uses the four-wave selection for any wave count.
	template <typename Problem>
	struct FourWaveTailCompV3 : ck_tile::GemmPipelineAgBgCrCompV3<Problem> {
		using CompV3 = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
		template <typename ADramWindow, typename BDramWindow>
		CK_TILE_DEVICE auto operator()(const ADramWindow& a_window, const BDramWindow& b_window, ck_tile::index_t num_loop, void* p_smem) const {
			const auto run = [&](auto has_hot_loop, auto tail_number) {
				return typename CompV3::template PipelineImpl<CompV3::Scheduler>{}.template operator()<has_hot_loop.value, tail_number.value>(
					ck_tile::make_tuple(a_window), ck_tile::element_wise::PassThrough{},
					ck_tile::make_tuple(b_window), ck_tile::element_wise::PassThrough{}, num_loop, p_smem);
			};
			if (ck_tile::amd_wave_read_first_lane(num_loop > CompV3::PrefetchStages)) {
				return run(ck_tile::true_type{}, ck_tile::constant<ck_tile::TailNumber::Odd>{});
			} else if (ck_tile::amd_wave_read_first_lane(num_loop == CompV3::PrefetchStages)) {
				return run(ck_tile::false_type{}, ck_tile::constant<ck_tile::TailNumber::Even>{});
			}
			return run(ck_tile::false_type{}, ck_tile::constant<ck_tile::TailNumber::Odd>{});
		}
	};

} // namespace at::hip::detail::ck_tile_group_gemm
