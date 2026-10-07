#include <ATen/native/hip/ck_tile_group_gemm.h>
#include <c10/hip/HIPStream.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <array>
#include <cstdint>

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

		using Strides = std::array<int64_t, 3>;


		// Fills one ck_tile::GemmTransKernelArg per group on the device, so offs is never read on the host.
		// Launch with one thread per group. Exactly one of M, N, K is -1 when that dimension is jagged
		// (2D x 3D: M, 3D x 2D: N, 2D x 2D: K); all are set for 3D x 3D. Ported from
		// prepare_grouped_gemm_data in ATen/native/cuda/GroupMMCommon.cuh.
		template <typename DataType>
		__global__ void prepare_grouped_gemm_data(
				const DataType* A,                          // base pointer of mat_a (2D (sumM, K) / (M, sumK), or 3D (G, M, K))
				const DataType* B,                          // base pointer of mat_b (2D (K, sumN) / (sumK, N), or 3D (G, K, N))
				DataType* output,                           // base pointer of out (2D (sumM, N), or 3D (G, M, N) / (G, M, sumN))
				ck_tile::GemmTransKernelArg<>* group_args,  // output: one entry per group, read by GroupedGemmKernel
				const int32_t* offs,                        // cumulative end offset of each group along the jagged dim; nullptr for 3D x 3D
				int32_t M,                                  // rows of each group's A and C, or -1 if jagged (2D x 3D)
				int32_t N,                                  // columns of each group's B and C, or -1 if jagged (3D x 2D)
				int32_t K,                                  // contraction size of each group, or -1 if jagged (2D x 2D)
				Strides tensor_StrideA,                     // mat_a.strides() in elements; only the first dim() entries are valid
				Strides tensor_StrideB,                     // mat_b.strides() in elements
				Strides tensor_StrideOutput,                // out.strides() in elements
				Strides tensor_ShapeA,                      // mat_a.sizes(); used to check offs stays inside A
				Strides tensor_ShapeB,                      // mat_b.sizes(); used to check offs stays inside B
				bool a_row_major,                           // true if A is layed out in row-major order
				bool b_row_major) {                         // true if B is layed out in row-major order

				
		}

		// Launches the ck_tile grouped GEMM for one dtype / layout / config combination.
		// group_args must already be filled by prepare_grouped_gemm_data on the same stream.
		template <typename DataType, typename ALayout, typename BLayout, typename Config>
		void run_ck_tile_group_gemm(
				const void* group_args,  // device array of group_count ck_tile::GemmTransKernelArg<>
				int32_t group_count,     // number of groups (entries in group_args)
				hipStream_t stream) {    // current HIP stream
			// 1. using Kernel = typename CkTileGroupGemm<DataType, ALayout, BLayout, Config>::Kernel;
			// 2. block = Kernel::BlockSize(); grid = min(Kernel::MaxOccupancyGridSize(stream_config), number of CUs)
			// 3. ck_tile::launch_kernel(stream_config, ck_tile::make_kernel<1>(Kernel{}, grid, block, 0,
			//        ck_tile::cast_pointer_to_constant_address_space(group_args), group_count))
		}

		// Per-dtype part of group_gemm_ck_tile: fills the per-group arguments on the device, then
		// picks the layout and tile config and launches the kernel.
		template <typename DataType>
		void launch_ck_tile_group_gemm(
				const at::Tensor& mat_a,             // A, already in a layout we instantiate
				const at::Tensor& mat_b,             // B, already in a layout we instantiate
				const std::optional<at::Tensor>& offs,
				at::Tensor& out,
				at::Tensor& group_args,              // kByte buffer of group_count * sizeof(ck_tile::GemmTransKernelArg<>)
				bool a_row_major,
				bool b_row_major,
				int32_t M,                           // -1 if jagged, see prepare_grouped_gemm_data
				int32_t N,
				int32_t K,
				int32_t group_count) {
			// 1. stream = c10::hip::getCurrentHIPStream()
			// 2. prepare_grouped_gemm_data<DataType><<<1, group_count, 0, stream>>>(...) with
			//    mat_a/mat_b/out data pointers, offs (or nullptr), M, N, K, strides, sizes, layouts
			// 3. Choose a config from the host-known N and K (and whether K is jagged)
			// 4. Branch on a_row_major / b_row_major -> RowMajor / ColumnMajor and call
			//    run_ck_tile_group_gemm<DataType, ALayout, BLayout, Config>(group_args.data_ptr(), group_count, stream)
		}
	}

	void group_gemm_ck_tile(
			const at::Tensor& mat_a,
			const at::Tensor& mat_b,
			const std::optional<at::Tensor>& offs,
			const std::optional<at::Tensor>& /*bias*/,
			at::Tensor& out) {
		// 1. Layouts: get transposed flags (check_valid_strides_and_return_transposed in
		//    ATen/native/GroupedMMUtils.h); .contiguous() any layout we don't instantiate.
		//    Read strides only from the tensors actually launched with.
		// 2. M, N, K, group_count: see GroupMM.cu (-1 marks the jagged dim)
		// 3. Host checks: group_count <= 1024; offs->size(0) == group_count when an operand is 2D;
		//    sizes and strides fit in int32 (ck_tile::index_t)
		// 4. group_args = at::empty({group_count * sizeof(ck_tile::GemmTransKernelArg<>)}, out.options().dtype(at::kByte))
		// 5. Switch on mat_a.scalar_type(): kBFloat16 -> ck_tile::bf16_t, kHalf -> ck_tile::half_t, kFloat -> float,
		//    and call launch_ck_tile_group_gemm<DataType>(...)
	}
}
