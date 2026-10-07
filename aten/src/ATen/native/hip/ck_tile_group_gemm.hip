#include <ATen/native/hip/ck_tile_group_gemm.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>

namespace at::hip::detail {
	namespace {
		using ck_tile::index_t;

		// Tile config for the ck_tile grouped GEMM kernels.
		// Vector widths of 0 let ck_tile pick the widest global loads/stores; a nonzero value fixes it
		// (e.g. 1 for N/K or group boundaries that are not aligned to the default vector width).
		template <index_t M_Tile_, index_t N_Tile_, index_t K_Tile_, // output tile per workgroup, K step per main-loop iteration
				index_t M_Warp_Tile_, index_t N_Warp_Tile_, index_t K_Warp_Tile_, // shape of one warp-level MFMA
				index_t M_Warp_, index_t N_Warp_, // waves per workgroup along M and N
				bool PadM_ = false, bool PadN_ = false, bool PadK_ = false, // allow sizes that are not tile multiples
				bool DoubleSmemBuffer_ = false,
				index_t VectorSizeAB_ = 0, index_t VectorSizeC_ = 0>
		struct CkTileGroupGemmConfig {
			static constexpr index_t M_Tile = M_Tile_;
			static constexpr index_t N_Tile = N_Tile_;
			static constexpr index_t K_Tile = K_Tile_;

			static constexpr index_t M_Warp_Tile = M_Warp_Tile_;
			static constexpr index_t N_Warp_Tile = N_Warp_Tile_;
			static constexpr index_t K_Warp_Tile = K_Warp_Tile_;

			static constexpr index_t M_Warp = M_Warp_;
			static constexpr index_t N_Warp = N_Warp_;
			static constexpr index_t K_Warp = 1;

			static constexpr bool PadM = PadM_;
			static constexpr bool PadN = PadN_;
			static constexpr bool PadK = PadK_;
			static constexpr bool DoubleSmemBuffer = DoubleSmemBuffer_;

			static constexpr bool FixedVectorSizeAB = VectorSizeAB_ != 0;
			static constexpr bool FixedVectorSizeC = VectorSizeC_ != 0;
			static constexpr index_t VectorSizeAB = FixedVectorSizeAB ? VectorSizeAB_ : 1;
			static constexpr index_t VectorSizeC = FixedVectorSizeC ? VectorSizeC_ : 1;

			// GemmSpatiallyLocalTilePartitioner: GroupNum matches the 8 XCDs of gfx942/gfx950.
			static constexpr index_t TilePartitionerGroupNum = 8;
			static constexpr index_t TilePartitionerM01 = 4;
		};

		template<typename DataType, typename ALayout, typename BLayout, typename Config>
		struct CkTileGroupGemm {
			using CLayout = ck_tile::tensor_layout::gemm::RowMajor;
			using AccDataType = float;
			using PassThrough = ck_tile::element_wise::PassThrough;

			using GemmShape = ck_tile::TileGemmShape<
				ck_tile::sequence<Config::M_Tile, Config::N_Tile, Config::K_Tile>,
				ck_tile::sequence<Config::M_Warp, Config::N_Warp, Config::K_Warp>,
				ck_tile::sequence<Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile>
			>;

			using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
				GemmShape, Config::TilePartitionerGroupNum, Config::TilePartitionerM01
			>;

			using GemmTraits = ck_tile::PersistentTileGemmUniversalTraits<
				Config::PadM, Config::PadN, Config::PadK, Config::DoubleSmemBuffer,
				ALayout, BLayout, CLayout
			>;

			using PipelineProblem = ck_tile::UniversalGemmPipelineProblem<
				DataType, DataType, AccDataType, GemmShape, GemmTraits,
				ck_tile::GemmPipelineScheduler::Intrawave,
				PassThrough, PassThrough, DataType, DataType,
				Config::FixedVectorSizeAB, Config::VectorSizeAB, Config::VectorSizeAB
			>;

			using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;

			using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
				DataType, DataType, ck_tile::tuple<>, AccDataType,
				DataType, ck_tile::tuple<>, CLayout, PassThrough,
				TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
				Config::M_Warp, Config::N_Warp,
				Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile,
				PipelineProblem::TransposeC,
				1 /* kNumWaveGroups */, Config::FixedVectorSizeC, Config::VectorSizeC>
			>;

			using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
		};


		// bf16/fp16 configs from AMD Primus-Turbo (MIT), used on gfx942 and gfx950.
		// Selected by N: N % 256 == 0, N % 128 == 0, otherwise the N-padded config.
		using Cfg16_256x256x64      = CkTileGroupGemmConfig<256, 256, 64, 32, 32, 16, 2, 2, false, false, false>;
		using Cfg16_256x128x64      = CkTileGroupGemmConfig<256, 128, 64, 32, 32, 16, 2, 2, false, false, false>;
		using Cfg16_256x128x64_PadN = CkTileGroupGemmConfig<256, 128, 64, 32, 32, 16, 2, 2, false, true, false>;

		// fp8/bf8 configs from Primus-Turbo. Primus runs these through its quantized runner
		// (row/col, tensor and blockwise scales), not the bf16/fp16 runner above.
		// gfx942: selected by N like the 16-bit configs; mixed fp8 x bf8 uses the 32x32x16 padded one,
		// blockwise (ABQuantGrouped) scaling uses 128x128x128.
		using Cfg8_Gfx942_256x256x128_32x32x32           = CkTileGroupGemmConfig<256, 256, 128, 32, 32, 32, 2, 2, false, false, false>;
		using Cfg8_Gfx942_256x128x128_32x32x32           = CkTileGroupGemmConfig<256, 128, 128, 32, 32, 32, 2, 2, false, false, false>;
		using Cfg8_Gfx942_256x128x128_32x32x32_PadN      = CkTileGroupGemmConfig<256, 128, 128, 32, 32, 32, 2, 2, false, true, false>;
		using Cfg8_Gfx942_256x256x128_32x32x16_PadN      = CkTileGroupGemmConfig<256, 256, 128, 32, 32, 16, 2, 2, false, true, false>;
		using Cfg8_Gfx942_128x128x128_32x32x32           = CkTileGroupGemmConfig<128, 128, 128, 32, 32, 32, 2, 2, false, false, false>;

		// gfx950: N % 256 == 0 -> 256x256x128, N % 128 == 0 -> 256x256x128_PadN, otherwise 128x128x128_32x32x64;
		// blockwise (ABQuantGrouped) scaling uses the 1x4-wave config.
		using Cfg8_Gfx950_256x256x128_16x16x128          = CkTileGroupGemmConfig<256, 256, 128, 16, 16, 128, 2, 2, false, false, false>;
		using Cfg8_Gfx950_256x256x128_16x16x128_PadN     = CkTileGroupGemmConfig<256, 256, 128, 16, 16, 128, 2, 2, false, true, false>;
		using Cfg8_Gfx950_128x128x128_32x32x64           = CkTileGroupGemmConfig<128, 128, 128, 32, 32, 64, 2, 2, false, false, false>;
		using Cfg8_Gfx950_128x128x128_16x16x128_1x4Waves = CkTileGroupGemmConfig<128, 128, 128, 16, 16, 128, 1, 4, false, false, false>;
	}
}
