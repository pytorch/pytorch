#include <ATen/native/hip/ck_tile_group_gemm.h>
#include <ATen/native/hip/ck_tile_group_gemm/four_wave_tail_comp_v3.h>
#include <ATen/native/hip/ck_tile_group_gemm/no_split_k_epilogue.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/GroupedMMUtils.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <ck_tile/core.hpp>
#include <ck_tile/host.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>

#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace at::hip::detail {
	namespace {
		using ck_tile::index_t;
		using ck_tile::core::arch::amdgcn_target_id;
		using ck_tile_group_gemm::FourWaveTailCompV3;
		using ck_tile_group_gemm::NoSplitKEpilogue;
		using RowMajor = ck_tile::tensor_layout::gemm::RowMajor;
		using ColMajor = ck_tile::tensor_layout::gemm::ColumnMajor;

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
			// Only a 16-bit C vector width of 1 needs NoSplitKEpilogue; every other config uses CShuffleEpilogue as is.
			static constexpr bool UseNoSplitKEpilogue = FixedVectorSizeC && VectorSizeC == 1;

			// GemmSpatiallyLocalTilePartitioner: GroupNum matches the 8 XCDs of gfx942/gfx950.
			static constexpr index_t TilePartitionerGroupNum = 8;
			static constexpr index_t TilePartitionerM01 = 4;
		};

		template<typename InputType, typename OutputType, typename ALayout, typename BLayout, typename Config>
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
				InputType, InputType, AccDataType, GemmShape, GemmTraits,
				ck_tile::GemmPipelineScheduler::Intrawave,
				PassThrough, PassThrough, InputType, InputType,
				Config::FixedVectorSizeAB, Config::VectorSizeAB, Config::VectorSizeAB
			>;

			// Eight-wave tiles avoid the 256-accumulator register limit of four-wave 256x256 tiles. CK's eight-wave
			// support has two bugs we work around: CompV3's tail selection (FourWaveTailCompV3) and CShuffleEpilogue,
			// which writes wrong results, so they store straight from the accumulators instead.
			static constexpr bool EightWaves = Config::M_Warp * Config::N_Warp == 8;

			using GemmPipeline = std::conditional_t<EightWaves,
				FourWaveTailCompV3<PipelineProblem>, ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>>;

			using DirectEpilogue = ck_tile::DefaultGemm2DEpilogue<ck_tile::DefaultGemm2DEpilogueProblem<
				InputType, InputType, ck_tile::tuple<>, AccDataType,
				OutputType, ck_tile::tuple<>, CLayout, PassThrough,
				TilePartitioner::MPerBlock, TilePartitioner::NPerBlock, Config::PadM, Config::PadN,
				Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile,
				PipelineProblem::TransposeC>>;

			using CShuffleEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
				InputType, InputType, ck_tile::tuple<>, AccDataType,
				OutputType, ck_tile::tuple<>, CLayout, PassThrough,
				TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
				Config::M_Warp, Config::N_Warp,
				Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile,
				PipelineProblem::TransposeC,
				1 /* kNumWaveGroups */, Config::FixedVectorSizeC, Config::VectorSizeC>
			>;
			using GemmEpilogue = std::conditional_t<EightWaves, DirectEpilogue,
				std::conditional_t<Config::UseNoSplitKEpilogue, NoSplitKEpilogue<CShuffleEpilogue>, CShuffleEpilogue>>;

			using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
		};

		// helper function to get the number of elements in one 16 byte load vector op
		template <typename T>
		constexpr index_t vector16_elems() { return 16 / sizeof(T); }

		// A candidate config with its time estimate for the persistent kernel, which runs one workgroup per CU and hands
		// out output tiles round-robin: rounds * (kPerTileUs + K steps per tile * kPerStepUs), with rounds = ceil(tiles / CUs).
		// The two constants are fitted per config to measured kernel times; only their ratios between candidates matter.
		template <typename Config_, double kPerTileUs_, double kPerStepUs_>
		struct TimedConfig {
			using Config = Config_;
			static constexpr double kPerTileUs = kPerTileUs_;
			static constexpr double kPerStepUs = kPerStepUs_;
		};

		template <typename C>
		concept TimedCandidate = requires {
			typename C::Config;
			{ C::kPerTileUs } -> std::convertible_to<double>;
			{ C::kPerStepUs } -> std::convertible_to<double>;
		};

		template <typename Tuple>
		concept CandidateList = []<size_t... I>(std::index_sequence<I...>) {
			return (TimedCandidate<std::tuple_element_t<I, Tuple>> && ...);
		}(std::make_index_sequence<std::tuple_size_v<Tuple>>{});

		// Configs for one GEMM type (forward, dX or dW). When every contiguous dim is 16-byte aligned, the fitting Candidate
		// with the lowest estimated time is launched; otherwise Unaligned, which pads M, N and K and uses 1-element C vectors.
		// A specialization missing a name fails its static_assert.
		template <typename G>
		concept GemmTypeConfigs = CandidateList<typename G::Candidates> && requires { typename G::Unaligned; };

		template <typename T>
		concept GroupGemmConfigs = GemmTypeConfigs<typename T::Forward> && GemmTypeConfigs<typename T::GradInput> &&
				GemmTypeConfigs<typename T::GradWeight> && requires { typename T::Fallback; };

		// Tile configs for one target. Deliberately undefined so an arch without configs fails to compile.
		// InputType is the type of both A and B: CK only has MFMAs for matching A and B types. The output type only
		// changes the epilogue's store, which every config handles, so it does not select configs.
		template <amdgcn_target_id Arch, typename InputType>
		struct CkTileGroupGemmConfigs;

		// configs for bf16 inputs on gfx950
		template <>
		struct CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t> {
			// Pads M, N and K with 1-element C vectors, so it accepts every shape.
			using AnyShape = CkTileGroupGemmConfig<256, 128, 64, 16, 16, 32, 2, 2, true, true, true, false, 0, 1>;

			// Time constants (us) fitted to standalone kernel timings of the Primus-Turbo grouped GEMM benchmark shapes at 256 CUs.
			// Smaller tiles win only when the large ones leave CUs idle or a mostly empty last round.

			// Forward (2D x 3D, 3D x 3D): row-major A, column-major B
			struct Forward {
				using Candidates = std::tuple<
					// 256x192 holds the 192 accumulators per lane this compiler keeps in VGPRs
					TimedConfig<CkTileGroupGemmConfig<256, 192, 64, 32, 32, 16, 2, 2, false, true, false>, 5.891, 1.218>,
					TimedConfig<CkTileGroupGemmConfig<256, 128, 64, 32, 32, 16, 2, 2, false, true, false>, 5.892, 0.9473>,
					TimedConfig<CkTileGroupGemmConfig<128, 192, 64, 32, 32, 16, 2, 2, false, true, false>, 3.790, 0.7594>,
					TimedConfig<CkTileGroupGemmConfig<128, 128, 64, 32, 32, 16, 2, 2, false, true, false>, 2.768, 0.5831>>;
				using Unaligned = AnyShape;
			};

			// dX (2D x 3D, 3D x 3D): row-major A and B
			struct GradInput {
				using Candidates = std::tuple<
					// eight waves split 256x256 into 128 accumulators per lane
					TimedConfig<CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, false, false, false>, 0.0, 0.9204>,
					TimedConfig<CkTileGroupGemmConfig<256, 128, 64, 16, 16, 32, 2, 2, false, false, false>, 7.261, 1.0627>,
					TimedConfig<CkTileGroupGemmConfig<256, 128, 64, 32, 32, 16, 2, 2, false, true, false>, 5.849, 1.2472>,
					TimedConfig<CkTileGroupGemmConfig<128, 128, 64, 32, 32, 16, 2, 2, false, true, false>, 4.575, 0.8568>>;
				using Unaligned = AnyShape;
			};

			// dW (2D x 2D): column-major A, row-major B, jagged K, so every config pads K
			struct GradWeight {
				using Candidates = std::tuple<
					TimedConfig<CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, false, false, true>, 5.724, 1.0234>,
					TimedConfig<CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, true, true, true>, 6.694, 1.1341>,
					TimedConfig<CkTileGroupGemmConfig<256, 128, 32, 32, 32, 16, 2, 2, true, true, true>, 6.160, 0.8104>>;
				using Unaligned = AnyShape;
			};

			// Layouts none of the three cover, e.g. column-major A with jagged M or jagged N (3D x 2D)
			using Fallback = AnyShape;
		};

		// fp16 has the same element size and MFMA shapes as bf16
		template <>
		struct CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::half_t>
			: CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t> {};

		static_assert(GroupGemmConfigs<CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t>>);
		static_assert(GroupGemmConfigs<CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::half_t>>);

		// Host-side view of one launch, for picking a config. -1 marks the jagged dim.
		struct GemmProblem {
			int64_t M, N, K;
			int64_t groups;
			int64_t jagged_total; // the jagged dim's total over all groups (e.g. all tokens); unused for 3D x 3D
			int64_t cus;          // compute units, one persistent workgroup each
			bool aligned;         // every contiguous dim of A, B and C is a whole number of 16-byte vectors
			bool fits_test_m;     // dW: M is an output dim and A's contiguous dim, so configs must fit it too
		};

		// A config fits when every known dim it does not pad is a multiple of its tile.
		template <typename Config>
		bool config_fits(const GemmProblem& p) {
			const auto fits = [](int64_t extent, index_t tile, bool pad) { return extent < 0 || pad || extent % tile == 0; };
			return fits(p.N, Config::N_Tile, Config::PadN) && fits(p.K, Config::K_Tile, Config::PadK) &&
					(!p.fits_test_m || fits(p.M, Config::M_Tile, Config::PadM));
		}

		// Estimated kernel time of a TimedConfig, see TimedConfig. A jagged M or N counts ceil(total / tile) tiles, exact
		// when every group is a whole number of tiles; a jagged K uses the average per group.
		template <TimedCandidate Timed>
		double estimated_us(const GemmProblem& p) {
			using Config = typename Timed::Config;
			const auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
			int64_t tiles = 0;
			if (p.M < 0) {
				tiles = ceil_div(p.jagged_total, Config::M_Tile) * ceil_div(p.N, Config::N_Tile);
			} else if (p.N < 0) {
				tiles = ceil_div(p.jagged_total, Config::N_Tile) * ceil_div(p.M, Config::M_Tile);
			} else {
				tiles = p.groups * ceil_div(p.M, Config::M_Tile) * ceil_div(p.N, Config::N_Tile);
			}
			const int64_t k = p.K < 0 ? ceil_div(p.jagged_total, p.groups) : p.K;
			return static_cast<double>(ceil_div(tiles, p.cus)) * (Timed::kPerTileUs + ceil_div(k, Config::K_Tile) * Timed::kPerStepUs);
		}

		template <GemmTypeConfigs Type, typename Run>
		void run_gemm_type(const GemmProblem& p, Run&& run) {
			using Candidates = typename Type::Candidates;
			constexpr size_t count = std::tuple_size_v<Candidates>;
			size_t best = count;
			if (p.aligned) {
				double best_us = std::numeric_limits<double>::infinity();
				[&]<size_t... I>(std::index_sequence<I...>) {
					const auto consider = [&](auto index, auto timed) {
						using Timed = decltype(timed);
						if (!config_fits<typename Timed::Config>(p)) {
							return;
						}
						const double us = estimated_us<Timed>(p);
						if (us < best_us) {
							best_us = us;
							best = index;
						}
					};
					(consider(I, std::tuple_element_t<I, Candidates>{}), ...);
				}(std::make_index_sequence<count>{});
			}
			if (best == count) {
				TORCH_CHECK(config_fits<typename Type::Unaligned>(p), "ck_tile group gemm: the Unaligned config must accept every shape");
				run(typename Type::Unaligned{});
				return;
			}
			[&]<size_t... I>(std::index_sequence<I...>) {
				((I == best ? run(typename std::tuple_element_t<I, Candidates>::Config{}) : void()), ...);
			}(std::make_index_sequence<count>{});
		}

		using Strides = std::array<int64_t, 3>;

		// dW (2D x 2D): GroupedGemmKernel skips groups with K == 0 (no tokens routed to the expert) and never writes their
		// output, whose correct value is zero. Zeroes those groups only, so the output can be allocated uninitialized.
		// One grid row per group; blocks of non-empty groups exit immediately.
		template <typename OutputType>
		__global__ void zero_empty_k_groups(
				OutputType* output,     // base pointer of out (G, M, N)
				const int32_t* offs,    // cumulative end offset of each group along K
				int32_t M,
				int32_t N,
				int64_t group_stride,   // out.stride(0)
				int64_t row_stride) {   // out.stride(1)
			const int32_t group = blockIdx.y;
			const int32_t start = group == 0 ? 0 : offs[group - 1];
			if (offs[group] != start) {
				return;
			}
			OutputType* group_out = output + group * group_stride;
			const int64_t size = static_cast<int64_t>(M) * N;
			for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < size;
					i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
				group_out[(i / N) * row_stride + i % N] = ck_tile::type_convert<OutputType>(0.0f);
			}
		}

		// Fills one ck_tile::GemmTransKernelArg per group on the device. 
		// Basically extracts A, B, and output ptrs and strides for one GEMM and puts in in the corresponding entry in the group_args array.
		// This is a kernel intended to be launched with one thread per group on the device so offs is never read on the host.
		template <typename InputType, typename OutputType>
		__global__ void prepare_grouped_gemm_data(
				const InputType* A,                         // base pointer of mat_a (2D (sumM, K) / (M, sumK), or 3D (G, M, K))
				const InputType* B,                         // base pointer of mat_b (2D (K, sumN) / (sumK, N), or 3D (G, K, N))
				OutputType* output,                         // base pointer of out (2D (sumM, N) / (M, sumN), or 3D (G, M, N))
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
			const int32_t thread_id = threadIdx.x;

			const InputType* a_ptr = nullptr; //pointer to the start of A[thread_id]
			const InputType* b_ptr = nullptr; //pointer to the start of B[thread_id]
			OutputType* output_ptr = nullptr; //pointer to the start of output[thread_id]
			int64_t lda = 0, ldb = 0, ldc = 0; //leading dimensions of A, B, and output

			// Start of this group along the jagged dim and its size; offs holds cumulative end offsets.
			int32_t start = 0;
			int32_t delta = 0;
			if (offs != nullptr) {
				start = thread_id == 0 ? 0 : offs[thread_id - 1];
				delta = offs[thread_id] - start;
				CUDA_KERNEL_ASSERT(delta >= 0 && "expected offs to be non-decreasing\n");
			}

			if (M < 0) { //M is the dynamic dimension being split, 2D x 3D
				// A (sumM, K), B (G, K, N), out (sumM, N): rows start..end of A and out, all of B[thread_id]
				CUDA_KERNEL_ASSERT(offs[thread_id] <= tensor_ShapeA[0] && "expected offs to stay inside mat_a\n");
				M = delta;
				a_ptr = A + start * tensor_StrideA[0]; // skip `start` rows, for either layout of A
				b_ptr = B + thread_id * tensor_StrideB[0];
				output_ptr = output + start * tensor_StrideOutput[0];

				lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
				ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
				ldc = tensor_StrideOutput[0]; // output is always row-major
			} else if (N < 0) { //N is the dynamic dimension being split, 3D x 2D
				// A (G, M, K), B (K, sumN), out (M, sumN): all of A[thread_id], columns start..end of B and out
				CUDA_KERNEL_ASSERT(offs[thread_id] <= tensor_ShapeB[1] && "expected offs to stay inside mat_b\n");
				N = delta;
				a_ptr = A + thread_id * tensor_StrideA[0];
				b_ptr = B + start * tensor_StrideB[1]; // skip `start` columns, for either layout of B
				output_ptr = output + start * tensor_StrideOutput[1];

				lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
				ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
				ldc = tensor_StrideOutput[0]; // output is ALWAYS row-major
			} else if (K < 0) { //K is the dynamic dimension being split, 2D x 2D
				// A (M, sumK), B (sumK, N), out (G, M, N): columns start..end of A, rows start..end of B, all of out[thread_id]
				CUDA_KERNEL_ASSERT(offs[thread_id] <= tensor_ShapeA[1] && offs[thread_id] <= tensor_ShapeB[0] && "expected offs to stay inside mat_a and mat_b\n");
				K = delta;
				a_ptr = A + start * tensor_StrideA[1]; // skip `start` columns of A
				b_ptr = B + start * tensor_StrideB[0]; // skip `start` rows of B
				output_ptr = output + thread_id * tensor_StrideOutput[0];

				lda = a_row_major ? tensor_StrideA[0] : tensor_StrideA[1];
				ldb = b_row_major ? tensor_StrideB[0] : tensor_StrideB[1];
				ldc = tensor_StrideOutput[1]; // output is ALWAYS row-major
			} else { //3D x 3D
				a_ptr = A + thread_id * tensor_StrideA[0]; //tensor_StrideA[0] should be M * K, ignoring padding, etc..
				b_ptr = B + thread_id * tensor_StrideB[0]; //tensor_StrideB[0] should be K * N, ignoring padding, etc..
				output_ptr = output + thread_id * tensor_StrideOutput[0]; //tensor_StrideOutput[0] should be M * N, ignoring padding, etc..

				lda = a_row_major ? tensor_StrideA[1] : tensor_StrideA[2];
				ldb = b_row_major ? tensor_StrideB[1] : tensor_StrideB[2];
				ldc = tensor_StrideOutput[1]; // output is ALWAYS row-major
			}

			auto& arg = group_args[thread_id];
			auto& karg = arg.group_karg;

			// looks sketch but it is a hack to set as_ptr and bs_ptr without going through the constructor
			const_cast<std::array<const void*, 1>&>(karg.as_ptr)[0] = a_ptr;
			const_cast<std::array<const void*, 1>&>(karg.bs_ptr)[0] = b_ptr;
			karg.e_ptr = output_ptr;

			karg.M = M;
			karg.N = N;
			karg.K = K;
			karg.stride_As[0] = lda;
			karg.stride_Bs[0] = ldb;
			karg.stride_E = ldc;
			karg.k_batch = 1;
			arg.block_start = 0;
			arg.block_end = 0;
		}

		// Launches the ck_tile grouped GEMM for one dtype / layout / config combination.
		// group_args must already be filled by prepare_grouped_gemm_data on the same stream.
		template <typename InputType, typename OutputType, typename ALayout, typename BLayout, typename Config>
		void run_ck_tile_group_gemm(
				const void* group_args,  // device array of group_count ck_tile::GemmTransKernelArg<>
				int32_t group_count,     // number of groups (entries in group_args)
				hipStream_t stream) {    // current HIP stream
			using Kernel = typename CkTileGroupGemm<InputType, OutputType, ALayout, BLayout, Config>::Kernel;

			// Persistent kernel: one workgroup per CU, each loops over the tiles of all groups.
			const dim3 grid(at::cuda::getCurrentDeviceProperties()->multiProcessorCount);
			const dim3 block = Kernel::BlockSize();

			ck_tile::launch_kernel(
					ck_tile::stream_config{stream},
					ck_tile::make_kernel<1>(Kernel{}, grid, block, 0,
							ck_tile::cast_pointer_to_constant_address_space(group_args), group_count));
			C10_CUDA_KERNEL_LAUNCH_CHECK();
		}

		// Per-dtype and per-arch part of group_gemm_ck_tile: fills the per-group arguments on the device, then
		// picks the layout and tile config and launches the kernel.
		template <amdgcn_target_id Arch, typename InputType, typename OutputType>
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
				int32_t group_count,
				int64_t jagged_total) {              // the jagged dim's total over all groups, see group_gemm_ck_tile
			const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

			auto to_strides = [](at::IntArrayRef values) {
				Strides result{};
				std::copy(values.begin(), values.end(), result.begin());
				return result;
			};

			// extract A, B, and output ptrs and strides for one GEMM and puts in in the corresponding entry in the group_args array.
			// we do this concurrently via the device launch, one thread per GEMM.
			prepare_grouped_gemm_data<InputType, OutputType><<<1, group_count, 0, stream>>>(
					reinterpret_cast<const InputType*>(mat_a.const_data_ptr()),
					reinterpret_cast<const InputType*>(mat_b.const_data_ptr()),
					reinterpret_cast<OutputType*>(out.data_ptr()),
					reinterpret_cast<ck_tile::GemmTransKernelArg<>*>(group_args.data_ptr()),
					offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
					M, N, K,
					to_strides(mat_a.strides()), to_strides(mat_b.strides()), to_strides(out.strides()),
					to_strides(mat_a.sizes()), to_strides(mat_b.sizes()),
					a_row_major, b_row_major);
			C10_CUDA_KERNEL_LAUNCH_CHECK();

			if (K < 0) {
				zero_empty_k_groups<OutputType><<<dim3(64, group_count), 256, 0, stream>>>(
						reinterpret_cast<OutputType*>(out.data_ptr()), offs->const_data_ptr<int32_t>(), M, N, out.stride(0), out.stride(1));
				C10_CUDA_KERNEL_LAUNCH_CHECK();
			}

			// Instantiates and launches the kernel for one config, picking the layout types at runtime.
			auto run_with_config = [&](auto config) {
				using Config = decltype(config);
				if (a_row_major && !b_row_major) {
					run_ck_tile_group_gemm<InputType, OutputType, RowMajor, ColMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else if (a_row_major && b_row_major) {
					run_ck_tile_group_gemm<InputType, OutputType, RowMajor, RowMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else if (!a_row_major && b_row_major) {
					run_ck_tile_group_gemm<InputType, OutputType, ColMajor, RowMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else {
					TORCH_CHECK(false, "ck_tile group gemm: column-major A with column-major B is not instantiated; make one of them contiguous");
				}
			};

			using Configs = CkTileGroupGemmConfigs<Arch, InputType>;
			// Contiguous dims: A's is K (row-major) or M, B's is N (row-major) or K, C's is N. A jagged dim (-1) is never contiguous.
			const auto vectors_fit = [](int64_t extent, index_t elems) { return extent < 0 || extent % elems == 0; };
			const GemmProblem problem{
				M, N, K, group_count, jagged_total, at::cuda::getCurrentDeviceProperties()->multiProcessorCount,
				vectors_fit(a_row_major ? K : M, vector16_elems<InputType>()) &&
						vectors_fit(b_row_major ? N : K, vector16_elems<InputType>()) &&
						vectors_fit(N, vector16_elems<OutputType>()),
				/*fits_test_m=*/K < 0};

			if (K < 0) {
				// dW (2D x 2D): K is jagged, M and N are fixed per group
				run_gemm_type<typename Configs::GradWeight>(problem, run_with_config);
			} else if (a_row_major && !b_row_major && N > 0) {
				// Forward (2D x 3D, 3D x 3D)
				run_gemm_type<typename Configs::Forward>(problem, [&](auto config) {
					run_ck_tile_group_gemm<InputType, OutputType, RowMajor, ColMajor, decltype(config)>(group_args.const_data_ptr(), group_count, stream);
				});
			} else if (a_row_major && b_row_major && N > 0) {
				// dX (2D x 3D, 3D x 3D)
				run_gemm_type<typename Configs::GradInput>(problem, [&](auto config) {
					run_ck_tile_group_gemm<InputType, OutputType, RowMajor, RowMajor, decltype(config)>(group_args.const_data_ptr(), group_count, stream);
				});
			} else {
				run_with_config(typename Configs::Fallback{});
			}
		}
	}

	void group_gemm_ck_tile(
			const at::Tensor& mat_a_in,
			const at::Tensor& mat_b_in,
			const std::optional<at::Tensor>& offs,
			const std::optional<at::Tensor>& /*bias*/,
			at::Tensor& out) {
		if (out.numel() == 0) {
			return;
		}

		const bool a_is_2d = mat_a_in.dim() == 2;
		const bool b_is_2d = mat_b_in.dim() == 2;

		// 1. Layouts. The jagged dim must not be the contiguous one: groups can start at any offset along it,
		// which breaks the vectorized loads/stores. Copy into a suitable layout only when needed.
		auto row_major = [](const at::Tensor& t) { return !at::native::check_valid_strides_and_return_transposed(t); };
		auto to_row_major = [](const at::Tensor& t) { return t.contiguous(); };
		auto to_col_major = [](const at::Tensor& t) { return t.transpose(-2, -1).contiguous().transpose(-2, -1); };
		at::Tensor mat_a = mat_a_in;
		at::Tensor mat_b = mat_b_in;
		if (a_is_2d && b_is_2d) {
			// jagged K: K must not be contiguous -> column-major A (M, sumK), row-major B (sumK, N)
			if (row_major(mat_a)) mat_a = to_col_major(mat_a);
			if (!row_major(mat_b)) mat_b = to_row_major(mat_b);
		} else if (a_is_2d) {
			// jagged M: M must not be contiguous -> row-major A (sumM, K); B can be either
			if (!row_major(mat_a)) mat_a = to_row_major(mat_a);
		} else if (b_is_2d) {
			// jagged N: N must not be contiguous in B -> column-major B (K, sumN); row-major A
			if (row_major(mat_b)) mat_b = to_col_major(mat_b);
			if (!row_major(mat_a)) mat_a = to_row_major(mat_a);
		} else if (!row_major(mat_a) && !row_major(mat_b)) {
			// 3D x 3D: every layout pair except column-major A with column-major B is instantiated
			mat_a = to_row_major(mat_a);
		}
		const bool a_row_major = row_major(mat_a);
		const bool b_row_major = row_major(mat_b);

		// 2. Per-group sizes; -1 marks the jagged dim (filled per group from offs on the device).
		int64_t M = mat_a.size(-2);
		int64_t K = mat_a.size(-1);
		int64_t N = mat_b.size(-1);
		int64_t group_count = 0;
		// Total of the jagged dim over all groups, known without reading offs; used to estimate each config's time.
		int64_t jagged_total = 0;
		if (a_is_2d && b_is_2d) {
			group_count = offs->size(0);
			jagged_total = K;
			K = -1;
		} else if (a_is_2d) {
			group_count = mat_b.size(0);
			jagged_total = M;
			M = -1;
		} else if (b_is_2d) {
			group_count = mat_a.size(0);
			jagged_total = N;
			N = -1;
		} else {
			group_count = mat_a.size(0);
		}

		// 3. Host checks.
		if (group_count == 0) {
			return;
		}
		if (K == 0) { 
			out.zero_(); return; 
		}

		TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(), "ck_tile group gemm: mat_a and mat_b must have the same dtype, got ",
				mat_a.scalar_type(), " and ", mat_b.scalar_type());
		TORCH_CHECK(group_count <= 1024, "ck_tile group gemm: at most 1024 groups are supported, got ", group_count);
		if (a_is_2d != b_is_2d) {
			TORCH_CHECK(offs->size(0) == group_count, "ck_tile group gemm: expected offs to have ", group_count,
					" entries (one per group), got ", offs->size(0));
		}
		// ck_tile stores sizes and leading dims as 32-bit index_t; per-group pointers are computed in 64 bits.
		constexpr int64_t index_max = std::numeric_limits<index_t>::max();
		for (const at::Tensor* t : {&mat_a, &mat_b, &out}) {
			for (int64_t d = 0; d < t->dim(); ++d) {
				TORCH_CHECK(t->size(d) <= index_max && (d == 0 || t->stride(d) <= index_max),
						"ck_tile group gemm: sizes and leading dimensions must fit in int32, got sizes ", t->sizes(),
						" strides ", t->strides());
			}
		}
		// ck_tile's vector loads need the host-known extent along each contiguous dim to be a multiple of 16 bytes.
		const int64_t vector_elems = 16 / mat_a.element_size();
		auto aligned = [&](int64_t extent) { return extent < 0 || extent % vector_elems == 0; };
		TORCH_CHECK_NOT_IMPLEMENTED(aligned(a_row_major ? K : M) && aligned(b_row_major ? N : K),
				"ck_tile group gemm: the contiguous dims of A and B must be multiples of ", vector_elems,
				" elements, got M=", M, " N=", N, " K=", K, " (-1 = jagged)");

		// 4. One ck_tile::GemmTransKernelArg per group, filled on the device by prepare_grouped_gemm_data.
		at::Tensor group_args = at::empty(
				{group_count * static_cast<int64_t>(sizeof(ck_tile::GemmTransKernelArg<>))},
				out.options().dtype(at::kByte));

		// 5. Arch and dtype dispatch. The arch is only known at runtime; integral_constant turns it into a template argument.
		// Only input types with a CkTileGroupGemmConfigs specialization are instantiated, each with every output type.
		auto with_ck_type = [](at::ScalarType dtype, auto&& f) {
			switch (dtype) {
				case at::kBFloat16: return f(ck_tile::bf16_t{});
				case at::kHalf: return f(ck_tile::half_t{});
				case at::kFloat: return f(float{});
				default: TORCH_CHECK(false, "ck_tile group gemm: unsupported dtype ", dtype);
			}
		};
		auto launch = [&](auto arch_tag) {
			with_ck_type(mat_a.scalar_type(), [&](auto input_tag) {
				constexpr amdgcn_target_id Arch = decltype(arch_tag)::value;
				using InputType = decltype(input_tag);
				if constexpr (GroupGemmConfigs<CkTileGroupGemmConfigs<Arch, InputType>>) {
					with_ck_type(out.scalar_type(), [&](auto output_tag) {
						launch_ck_tile_group_gemm<Arch, InputType, decltype(output_tag)>(
								mat_a, mat_b, offs, out, group_args, a_row_major, b_row_major,
								static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(group_count), jagged_total);
					});
				} else {
					TORCH_CHECK_NOT_IMPLEMENTED(false, "ck_tile group gemm: no configs for ", mat_a.scalar_type(), " inputs on ",
							at::cuda::getCurrentDeviceProperties()->gcnArchName);
				}
			});
		};
		if (at::detail::getCUDAHooks().isGPUArch({"gfx950"})) {
			launch(std::integral_constant<amdgcn_target_id, amdgcn_target_id::GFX950>{});
		} else {
			TORCH_CHECK_NOT_IMPLEMENTED(false, "ck_tile group gemm: no tile configs for ", at::cuda::getCurrentDeviceProperties()->gcnArchName);
		}
	}
}
