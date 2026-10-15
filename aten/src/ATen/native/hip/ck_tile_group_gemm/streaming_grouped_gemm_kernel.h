#pragma once

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <cstdint>
#include <type_traits>

namespace at::hip::detail::ck_tile_group_gemm {

using ck_tile::index_t;
// CK_TILE_CONSTANT_ADDRESS_SPACE names these unqualified
using ck_tile::address_space_enum;
using ck_tile::safe_underlying_type_t;

	// Persistent grouped GEMM that runs CompV3's schedule (two global prefetch stages, one LDS stage) continuously
	// over every K step of every tile a workgroup owns: the global loads for the next tile are issued while the
	// current one finishes, so there is no per-tile prologue or pipeline drain. With StageOutput, a finished tile is
	// converted to C's type in a second LDS region, outside the main loop's buffer, and written out with 16-byte
	// stores (needs N % 8 == 0 and both LDS regions to fit); otherwise Epilogue stores it without using LDS.
	// Requires K_Tile | K.
	template <typename TilePartitioner, typename Problem, typename Epilogue, bool StageOutput>
	struct StreamingGroupedGemmKernel
			: ck_tile::GroupedGemmKernel<TilePartitioner, ck_tile::GemmPipelineAgBgCrCompV3<Problem>, Epilogue> {
		using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, ck_tile::GemmPipelineAgBgCrCompV3<Problem>, Epilogue>;
		using UniversalKernel = typename Kernel::Base;
		using CompV3 = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
		using Policy = ck_tile::GemmPipelineAgBgCrDefaultPolicy;
		using ImplBase = ck_tile::GemmPipelineAgBgCrImplBase<Problem, Policy>;
		using BlockGemm = typename CompV3::BlockGemm;
		using ADataType = typename CompV3::ADataType;
		using BDataType = typename CompV3::BDataType;
		using CDataType = typename Kernel::CDataType;
		using GroupArg = ck_tile::GemmTransKernelArg<>;

		static constexpr index_t MPerBlock = CompV3::MPerBlock;
		static constexpr index_t NPerBlock = CompV3::NPerBlock;
		static constexpr index_t KPerBlock = CompV3::KPerBlock;
		static constexpr index_t kBlockSize = CompV3::BlockSize;
		static constexpr bool is_a_col_major = std::is_same_v<typename CompV3::ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>;
		static constexpr bool is_b_row_major = std::is_same_v<typename CompV3::BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
		static constexpr bool a_tr = ImplBase::is_a_load_tr;
		static constexpr bool b_tr = ImplBase::is_b_load_tr;

		static constexpr index_t kVec = 16 / sizeof(CDataType);
		static constexpr index_t kOutPitch = NPerBlock + kVec; // LDS output row pitch, padded by 16 bytes against bank conflicts
		static constexpr index_t kVecPerRow = NPerBlock / kVec;
		static constexpr index_t kStoresPerThread = MPerBlock * kVecPerRow / kBlockSize;
		static constexpr index_t kMainSmem = CompV3::GetSmemSize();
		static constexpr index_t kOutSmem = StageOutput ? MPerBlock * kOutPitch * sizeof(CDataType) : 0;
		static_assert(!StageOutput || (MPerBlock * kVecPerRow % kBlockSize == 0 && kMainSmem % 16 == 0));
		static_assert(kMainSmem + kOutSmem <= 160 * 1024, "the main loop and output LDS regions must fit together");

		struct Tile {
			index_t group = -1, block_start = 0, block_end = 0, block_id;
			index_t i_m, i_n, num_k, k;
		};

		// Moves t to the group holding t.block_id, skipping K == 0 groups like GroupedGemmKernel, and sets its origin.
		CK_TILE_DEVICE static bool Seek(const GroupArg* args, index_t group_count, index_t grid_size, Tile& t) {
			while (t.block_id >= t.block_end || args[t.group].group_karg.K == 0) {
				if (t.group >= 0 && args[t.group].group_karg.K == 0) {
					while (t.block_id < t.block_end) t.block_id += grid_size;
				}
				if (++t.group >= group_count) return false;
				t.block_start = t.block_end;
				t.block_end += TilePartitioner::GridSize(args[t.group].group_karg.M, args[t.group].group_karg.N);
			}
			const auto& ka = args[t.group].group_karg;
			const auto [iM, iN] = ck_tile::OffsettedTile1DPartitioner<TilePartitioner>::GetOffsetedTileIndex(
					0, ka.M, ka.N, t.block_id - t.block_start);
			t.i_m = ck_tile::amd_wave_read_first_lane(iM * MPerBlock);
			t.i_n = ck_tile::amd_wave_read_first_lane(iN * NPerBlock);
			t.num_k = ck_tile::amd_wave_read_first_lane(TilePartitioner::GetLoopNum(ka.K));
			t.k = 0;
			return true;
		}

		CK_TILE_DEVICE static auto MakeABlockWindow(const GroupArg* args, const Tile& t) {
			const auto& ka = args[t.group].group_karg;
			return UniversalKernel::MakeABlockWindows({static_cast<const ADataType*>(ka.as_ptr[0])}, ka, ka.K, t.i_m)
					.at(ck_tile::number<0>{});
		}
		CK_TILE_DEVICE static auto MakeBBlockWindow(const GroupArg* args, const Tile& t) {
			const auto& ka = args[t.group].group_karg;
			return UniversalKernel::MakeBBlockWindows({static_cast<const BDataType*>(ka.bs_ptr[0])}, ka, ka.K, t.i_n)
					.at(ck_tile::number<0>{});
		}

		CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* group_args_const, const index_t group_count) const {
			using namespace ck_tile;
			const GroupArg* args = reinterpret_cast<const GroupArg*>(cast_pointer_to_generic_address_space(group_args_const));
			const index_t grid_size = get_grid_size();
			
			// ct is the tile being computed; lt is where the global loads are, two K steps ahead
			Tile ct;
			ct.block_id = get_block_1d_id();
			if (!Seek(args, group_count, grid_size, ct)) return;
			Tile lt = ct;
			bool lt_valid = true;

			// shared memory in LDS. accessable by all the threads in the workgroup.
			__shared__ char smem[kMainSmem + kOutSmem];

			ImplBase base;
			auto&& [a_lds, b_lds] = base.GetABLdsTensorViews(smem);
			constexpr auto a_lds_load_distr = make_static_tile_distribution(BlockGemm::MakeABlockDistributionEncode());
			constexpr auto b_lds_load_distr = make_static_tile_distribution(BlockGemm::MakeBBlockDistributionEncode());
			auto&& [a_dram, a_copy_lds, a_gemm_lds] = base.GetAWindows(MakeABlockWindow(args, lt), a_lds, a_lds_load_distr);
			auto&& [b_dram, b_copy_lds, b_gemm_lds] = base.GetBWindows(MakeBBlockWindow(args, lt), b_lds, b_lds_load_distr);
			constexpr auto a_step = is_a_col_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);
			constexpr auto b_step = is_b_row_major ? make_array(KPerBlock, 0) : make_array(0, KPerBlock);

			auto a_global = load_tile(a_dram);
			auto b_global = load_tile(b_dram);
			constexpr auto a_store_dstr = []() {
				if constexpr (is_a_col_major && !a_tr) {
					return decltype(make_static_distributed_tensor<ADataType>(
							Policy::template MakeShuffledARegTileDistribution<Problem>()))::get_tile_distribution();
				} else {
					return decltype(a_global)::get_tile_distribution();
				}
			}();
			constexpr auto b_store_dstr = []() {
				if constexpr (is_b_row_major && !b_tr) {
					return decltype(make_static_distributed_tensor<BDataType>(
							Policy::template MakeShuffledBRegTileDistribution<Problem>()))::get_tile_distribution();
				} else {
					return decltype(b_global)::get_tile_distribution();
				}
			}();
			auto a_store = ImplBase::MakeDistributedLdsStoreWindow(a_copy_lds, a_store_dstr);
			auto b_store = ImplBase::MakeDistributedLdsStoreWindow(b_copy_lds, b_store_dstr);

			auto global_load = [&]() {
				a_global = load_tile(a_dram);
				b_global = load_tile(b_dram);
				move_tile_window(a_dram, a_step);
				move_tile_window(b_dram, b_step);
				++lt.k;
			};
			// Past the last tile the windows keep moving along K; those loads are never consumed.
			auto next_load_tile = [&]() {
				if (lt_valid && lt.k >= lt.num_k) {
					lt.block_id += grid_size;
					lt_valid = Seek(args, group_count, grid_size, lt);
					if (lt_valid) {
						a_dram = base.CopyADramWindow(MakeABlockWindow(args, lt));
						b_dram = base.CopyBDramWindow(MakeBBlockWindow(args, lt));
					}
				}
			};
			auto local_prefill = [&]() {
				block_sync_lds();
				if constexpr (is_a_col_major && !a_tr) {
					auto shuffled = make_static_distributed_tensor<ADataType>(Policy::template MakeShuffledARegTileDistribution<Problem>());
					transpose_tile2d(shuffled, a_global);
					base.LocalStore(a_store, shuffled);
				} else {
					base.LocalStore(a_store, a_global);
				}
				if constexpr (is_b_row_major && !b_tr) {
					auto shuffled = make_static_distributed_tensor<BDataType>(Policy::template MakeShuffledBRegTileDistribution<Problem>());
					transpose_tile2d(shuffled, b_global);
					base.LocalStore(b_store, shuffled);
				} else {
					base.LocalStore(b_store, b_global);
				}
			};

			BlockGemm block_gemm;
			auto c = block_gemm.MakeCBlockTile();
			tile_elementwise_inout([](auto& v) { v = 0; }, c);

			CDataType* out_lds = reinterpret_cast<CDataType*>(smem + kMainSmem);
			auto out_lds_window = make_tile_window(
					make_tensor_view<address_space_enum::lds>(out_lds,
							make_naive_tensor_descriptor(make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
									make_tuple(number<kOutPitch>{}, number<1>{}), number<kVec>{}, number<1>{})),
					make_tuple(number<MPerBlock>{}, number<NPerBlock>{}), {0, 0}, c.get_tile_distribution());
			const index_t tid = get_thread_id();
			// Writes the tile in LDS to C: consecutive threads store consecutive 16 bytes of a row. Rows past M fall
			// outside the buffer range; columns past N get an out-of-range offset.
			auto store_output = [&]() {
				const auto& ka = args[ct.group].group_karg;
				CDataType* c_ptr = static_cast<CDataType*>(ka.e_ptr) + static_cast<int64_t>(ct.i_m) * ka.stride_E + ct.i_n;
				const int64_t bytes = (static_cast<int64_t>(ka.M - ct.i_m) * ka.stride_E - ct.i_n) * sizeof(CDataType);
				const int32x4_t rsrc = make_wave_buffer_resource(c_ptr, static_cast<uint32_t>(bytes < 0xffffffff ? bytes : 0xffffffff));
				static_for<0, kStoresPerThread, 1>{}([&](auto j) {
					const index_t v = j.value * kBlockSize + tid;
					const index_t row = v / kVecPerRow;
					const index_t col = (v - row * kVecPerRow) * kVec;
					const int32x4_t val = *reinterpret_cast<const int32x4_t*>(out_lds + row * kOutPitch + col);
					const index_t offset = col < ka.N - ct.i_n ? (row * ka.stride_E + col) * index_t(sizeof(CDataType)) : index_t(0x80000000);
					llvm_amdgcn_raw_buffer_store_i32x4(val, rsrc, offset, 0, 0);
				});
			};

			// Prologue: K step 0 in LDS and in the fragments, step 1 in registers.
			global_load();
			next_load_tile();
			local_prefill();
			global_load();
			next_load_tile();
			block_sync_lds();
			block_gemm.LocalPrefetch(a_gemm_lds, b_gemm_lds, bool_constant<a_tr>{}, bool_constant<b_tr>{});
			__builtin_amdgcn_sched_barrier(0);

			do {
				for (index_t k = 0; k < ct.num_k - 1; ++k) {
					local_prefill();
					global_load();
					block_gemm(c, a_gemm_lds, b_gemm_lds);
					block_sync_lds();
					block_gemm.LocalPrefetch(a_gemm_lds, b_gemm_lds, bool_constant<a_tr>{}, bool_constant<b_tr>{});
					CompV3::template PipelineImpl<GemmPipelineScheduler::Intrawave>::HotLoopScheduler();
					next_load_tile();
				}
				// Last K step of the tile; the output is stored before the next tile's first fragments are read.
				local_prefill();
				global_load();
				block_gemm(c, a_gemm_lds, b_gemm_lds);
				__builtin_amdgcn_sched_barrier(0);
				if constexpr (StageOutput) {
					store_tile(out_lds_window, cast_tile<CDataType>(c));
					block_sync_lds();
					store_output();
				} else {
					const auto& ka = args[ct.group].group_karg;
					auto c_window = UniversalKernel::template MakeCBlockWindows<memory_operation_enum::set>(
							static_cast<CDataType*>(ka.e_ptr), ka, ct.i_m, ct.i_n);
					Epilogue{}(c_window, c, UniversalKernel::MakeDBlockWindows(ka.ds_ptr, ka, ct.i_m, ct.i_n), nullptr);
					block_sync_lds();
				}
				tile_elementwise_inout([](auto& v) { v = 0; }, c);
				__builtin_amdgcn_sched_barrier(0);
				block_gemm.LocalPrefetch(a_gemm_lds, b_gemm_lds, bool_constant<a_tr>{}, bool_constant<b_tr>{});
				next_load_tile();
				ct.block_id += grid_size;
			} while (Seek(args, group_count, grid_size, ct));
		}
	};

} // namespace at::hip::detail::ck_tile_group_gemm
