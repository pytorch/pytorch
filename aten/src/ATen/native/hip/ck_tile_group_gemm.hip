#include <ATen/native/hip/ck_tile_group_gemm.h>
#include <ATen/native/hip/ck_tile_group_gemm/four_wave_tail_comp_v3.h>
#include <ATen/native/hip/ck_tile_group_gemm/streaming_grouped_gemm_kernel.h>
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
#include <type_traits>

namespace at::hip::detail {
	namespace {
		using ck_tile::index_t;
		using ck_tile::core::arch::amdgcn_target_id;
		using ck_tile_group_gemm::FourWaveTailCompV3;
		using ck_tile_group_gemm::StreamingGroupedGemmKernel;
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
				index_t VectorSizeAB_ = 0, index_t VectorSizeC_ = 0,
				bool Streaming_ = false> // StreamingGroupedGemmKernel instead of GroupedGemmKernel
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
			static constexpr bool Streaming = Streaming_;

			// GemmSpatiallyLocalTilePartitioner: GroupNum matches the 8 XCDs of gfx942/gfx950.
			static constexpr index_t TilePartitionerGroupNum = 8;
			static constexpr index_t TilePartitionerM01 = 4;
		};

		// GroupedGemmKernel instantiates a split-K (k_batch > 1) atomic_add epilogue next to the plain store one.
		// We always run k_batch == 1, so compile the atomic_add path out: 16-bit atomics reject a C vector width of 1.
		template <typename Epilogue>
		struct NoSplitKEpilogue : Epilogue {
			template <typename OutWindow, typename... Args>
			CK_TILE_DEVICE auto operator()(OutWindow& out_window, Args&&... args) {
				if constexpr (OutWindow::BottomTensorView::DstInMemOp == ck_tile::memory_operation_enum::set) {
					return Epilogue::operator()(out_window, std::forward<Args>(args)...);
				}
			}
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

			// Eight-wave tiles avoid the 256-accumulator register limit of four-wave 256x256 tiles. CK's eight-wave
			// support has two bugs we work around: CompV3's tail selection (FourWaveTailCompV3) and CShuffleEpilogue,
			// which writes wrong results, so they store straight from the accumulators instead.
			static constexpr bool EightWaves = Config::M_Warp * Config::N_Warp == 8;

			using GemmPipeline = std::conditional_t<EightWaves,
				FourWaveTailCompV3<PipelineProblem>, ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>>;

			using DirectEpilogue = ck_tile::DefaultGemm2DEpilogue<ck_tile::DefaultGemm2DEpilogueProblem<
				DataType, DataType, ck_tile::tuple<>, AccDataType,
				DataType, ck_tile::tuple<>, CLayout, PassThrough,
				TilePartitioner::MPerBlock, TilePartitioner::NPerBlock, Config::PadM, Config::PadN,
				Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile,
				PipelineProblem::TransposeC>>;

			using CShuffleEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
				DataType, DataType, ck_tile::tuple<>, AccDataType,
				DataType, ck_tile::tuple<>, CLayout, PassThrough,
				TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,
				Config::M_Warp, Config::N_Warp,
				Config::M_Warp_Tile, Config::N_Warp_Tile, Config::K_Warp_Tile,
				PipelineProblem::TransposeC,
				1 /* kNumWaveGroups */, Config::FixedVectorSizeC, Config::VectorSizeC>
			>;
			using GemmEpilogue = std::conditional_t<EightWaves, DirectEpilogue,
				std::conditional_t<Config::UseNoSplitKEpilogue, NoSplitKEpilogue<CShuffleEpilogue>, CShuffleEpilogue>>;

			using Kernel = std::conditional_t<Config::Streaming,
				StreamingGroupedGemmKernel<TilePartitioner, PipelineProblem, GemmEpilogue, !EightWaves>,
				ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>>;
		};

		// helper function to get the number of elements in one 16 byte load vector op
		template <typename T>
		constexpr index_t vector16_elems() { return 16 / sizeof(T); }

		// One size set, tried in this order: the first class whose condition holds and whose config fits is launched.
		//   N256 / N128 / N64: output dims divisible by 256 / 128 / 64 (dW: both M and N), so a tile that wide fits exactly
		//   Aligned:           every contiguous dim of A, B and C is a whole number of 16-byte vectors
		//   Unaligned:         any shape; must pad M, N and K and use 1-element C vectors
		// A specialization missing a name fails its static_assert.
		template <typename S>
		concept ConfigSizeSet = requires {
			typename S::N256;
			typename S::N128;
			typename S::N64;
			typename S::Aligned;
			typename S::Unaligned;
		};

		// One GEMM type (forward, dX or dW). FewTokens is used when the average tokens per expert is below
		// kFewTokensBelow; it is the slot for shorter tiles, which waste less of each expert's last partial tile.
		template <typename G>
		concept GemmTypeConfigs = requires {
			{ G::kFewTokensBelow } -> std::convertible_to<int64_t>;
		} && ConfigSizeSet<typename G::Main> && ConfigSizeSet<typename G::FewTokens>;

		template <typename T>
		concept GroupGemmConfigs = GemmTypeConfigs<typename T::Forward> && GemmTypeConfigs<typename T::GradInput> &&
				GemmTypeConfigs<typename T::GradWeight> && requires { typename T::Fallback; };

		// Tile configs for one target. Deliberately undefined so an arch without configs fails to compile.
		template <amdgcn_target_id Arch, typename AType, typename BType, typename OutputType>
        struct CkTileGroupGemmConfigs;

		// configs for a=bf16, b=bf16, c=bf16 for gfx950
		// this is re-used for any combo of a=bf16/fp16, b=bf16/fp16, c=bf16/fp16 on gfx950
		template <>
		struct CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t> {
			// Pads M, N and K with 1-element C vectors, so it accepts every shape.
			using AnyShape = CkTileGroupGemmConfig<256, 128, 64, 16, 16, 32, 2, 2, true, true, true, false, 0, 1>;

			// Forward (2D x 3D, 3D x 3D): row-major A, column-major B
			struct Forward {
				static constexpr int64_t kFewTokensBelow = 0;
				struct Main {
					// 256x192 holds the 192 accumulators per lane this compiler keeps in VGPRs; streaming removes per-tile drains
					using N256 = CkTileGroupGemmConfig<256, 192, 64, 32, 32, 16, 2, 2, false, true, false, false, 0, 0, true>;
					using N128 = N256;
					using N64 = N256;
					using Aligned = N256;
					using Unaligned = AnyShape;
				};
				using FewTokens = Main;
			};

			// dX (2D x 3D, 3D x 3D): row-major A and B
			struct GradInput {
				static constexpr int64_t kFewTokensBelow = 0;
				struct Main {
					// eight waves split 256x256 into 128 accumulators per lane
					using N256 = CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, false, false, false, false, 0, 0, true>;
					using N128 = CkTileGroupGemmConfig<256, 128, 64, 16, 16, 32, 2, 2, false, false, false>;
					using N64 = CkTileGroupGemmConfig<256, 128, 64, 32, 32, 16, 2, 2, false, true, false>;
					using Aligned = N64;
					using Unaligned = AnyShape;
				};
				using FewTokens = Main;
			};

			// dW (2D x 2D): column-major A, row-major B, jagged K, so every config pads K
			struct GradWeight {
				static constexpr int64_t kFewTokensBelow = 0;
				struct Main {
					using N256 = CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, false, false, true>;
					using N128 = CkTileGroupGemmConfig<256, 256, 32, 32, 32, 16, 2, 4, true, true, true>;
					using N64 = N128;
					using Aligned = N128;
					using Unaligned = AnyShape;
				};
				using FewTokens = Main;
			};

			// Layouts none of the three cover, e.g. column-major A with jagged M or jagged N (3D x 2D)
			using Fallback = AnyShape;
		};

		// fp16 has the same element size and MFMA shapes as bf16
		template <>
		struct CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>
			: CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t> {};

		static_assert(GroupGemmConfigs<CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t>>);
		static_assert(GroupGemmConfigs<CkTileGroupGemmConfigs<amdgcn_target_id::GFX950, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>>);

		// Host-side view of one launch, for picking a config. -1 marks the jagged dim.
		struct GemmProblem {
			int64_t M, N, K;
			int64_t avg_tokens;  // average tokens per expert: the jagged dim's total / groups, or M for 3D x 3D
			bool aligned;        // every contiguous dim of A, B and C is a whole number of 16-byte vectors
			bool classes_test_m; // dW: M is an output dim and A's contiguous dim, so the classes test it too
		};

		// A config fits when every known dim it does not pad is a multiple of its tile. Checked against the config itself,
		// so a wrong alias falls through to the next class instead of launching an out-of-bounds kernel.
		template <typename Config>
		bool config_fits(const GemmProblem& p) {
			const auto fits = [](int64_t extent, index_t tile, bool pad) { return extent < 0 || pad || extent % tile == 0; };
			return fits(p.N, Config::N_Tile, Config::PadN) && fits(p.K, Config::K_Tile, Config::PadK) &&
					(!p.classes_test_m || fits(p.M, Config::M_Tile, Config::PadM));
		}

		template <GemmTypeConfigs Type, typename Run>
		void run_gemm_type(const GemmProblem& p, Run&& run) {
			const auto run_size_set = [&](auto size_set) { // runs either the FewTokens or the Main config
				using Set = decltype(size_set);
				const auto divisible = [&](int64_t d) { return p.aligned && p.N % d == 0 && (!p.classes_test_m || p.M % d == 0); };
				const auto try_run = [&](auto config) {
					if (!config_fits<decltype(config)>(p)) {
						return false;
					}
					run(config);
					return true;
				};
				if (divisible(256) && try_run(typename Set::N256{})) return;
				if (divisible(128) && try_run(typename Set::N128{})) return;
				if (divisible(64) && try_run(typename Set::N64{})) return;
				if (p.aligned && try_run(typename Set::Aligned{})) return;
				TORCH_CHECK(try_run(typename Set::Unaligned{}), "ck_tile group gemm: the Unaligned config must accept every shape");
			};
			if (p.avg_tokens < Type::kFewTokensBelow) {
				run_size_set(typename Type::FewTokens{});
			} else {
				run_size_set(typename Type::Main{});
			}
		}

		using Strides = std::array<int64_t, 3>;

		// Fills one ck_tile::GemmTransKernelArg per group on the device. 
		// Basically extracts A, B, and output ptrs and strides for one GEMM and puts in in the corresponding entry in the group_args array.
		// This is a kernel intended to be launched with one thread per group on the device so offs is never read on the host.
		template <typename DataType>
		__global__ void prepare_grouped_gemm_data(
				const DataType* A,                          // base pointer of mat_a (2D (sumM, K) / (M, sumK), or 3D (G, M, K))
				const DataType* B,                          // base pointer of mat_b (2D (K, sumN) / (sumK, N), or 3D (G, K, N))
				DataType* output,                           // base pointer of out (2D (sumM, N) / (M, sumN), or 3D (G, M, N))
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

			const DataType* a_ptr = nullptr; //pointer to the start of A[thread_id]
			const DataType* b_ptr = nullptr; //pointer to the start of B[thread_id]
			DataType* output_ptr = nullptr; //pointer to the start of output[thread_id]
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
		template <typename DataType, typename ALayout, typename BLayout, typename Config>
		void run_ck_tile_group_gemm(
				const void* group_args,  // device array of group_count ck_tile::GemmTransKernelArg<>
				int32_t group_count,     // number of groups (entries in group_args)
				hipStream_t stream) {    // current HIP stream
			using Kernel = typename CkTileGroupGemm<DataType, ALayout, BLayout, Config>::Kernel;

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
		template <amdgcn_target_id Arch, typename DataType>
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
				int64_t avg_tokens) {                // average tokens per expert, see group_gemm_ck_tile
			//TODO: not a permanent restriction, will add support for mixed dtypes later
			TORCH_CHECK(mat_a.scalar_type() == mat_b.scalar_type(), "ck_tile group gemm: mixed dtypes are not supported, got mat_a ", mat_a.scalar_type(), " and mat_b ", mat_b.scalar_type());
			
			const hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

			auto to_strides = [](at::IntArrayRef values) {
				Strides result{};
				std::copy(values.begin(), values.end(), result.begin());
				return result;
			};

			// extract A, B, and output ptrs and strides for one GEMM and puts in in the corresponding entry in the group_args array.
			// we do this concurrently via the device launch, one thread per GEMM.
			prepare_grouped_gemm_data<DataType><<<1, group_count, 0, stream>>>(
					reinterpret_cast<const DataType*>(mat_a.const_data_ptr()),
					reinterpret_cast<const DataType*>(mat_b.const_data_ptr()),
					reinterpret_cast<DataType*>(out.data_ptr()),
					reinterpret_cast<ck_tile::GemmTransKernelArg<>*>(group_args.data_ptr()),
					offs.has_value() ? offs->const_data_ptr<int32_t>() : nullptr,
					M, N, K,
					to_strides(mat_a.strides()), to_strides(mat_b.strides()), to_strides(out.strides()),
					to_strides(mat_a.sizes()), to_strides(mat_b.sizes()),
					a_row_major, b_row_major);
			C10_CUDA_KERNEL_LAUNCH_CHECK();

			// Instantiates and launches the kernel for one config, picking the layout types at runtime.
			auto run_with_config = [&](auto config) {
				using Config = decltype(config);
				if (a_row_major && !b_row_major) {
					run_ck_tile_group_gemm<DataType, RowMajor, ColMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else if (a_row_major && b_row_major) {
					run_ck_tile_group_gemm<DataType, RowMajor, RowMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else if (!a_row_major && b_row_major) {
					run_ck_tile_group_gemm<DataType, ColMajor, RowMajor, Config>(group_args.const_data_ptr(), group_count, stream);
				} else {
					TORCH_CHECK(false, "ck_tile group gemm: column-major A with column-major B is not instantiated; make one of them contiguous");
				}
			};

			// TODO: add fp32 config
			TORCH_CHECK((!std::is_same_v<DataType, float>), "ck_tile group gemm: no fp32 config yet");

			using Configs = CkTileGroupGemmConfigs<Arch, DataType, DataType, DataType>;
			// Contiguous dims: A's is K (row-major) or M, B's is N (row-major) or K, C's is N. A jagged dim (-1) is never contiguous.
			const auto vectors_fit = [](int64_t extent, index_t elems) { return extent < 0 || extent % elems == 0; };
			const GemmProblem problem{
				M, N, K, avg_tokens,
				vectors_fit(a_row_major ? K : M, vector16_elems<DataType>()) &&
						vectors_fit(b_row_major ? N : K, vector16_elems<DataType>()) &&
						vectors_fit(N, vector16_elems<DataType>()),
				/*classes_test_m=*/K < 0};

			if (K < 0) {
				// dW (2D x 2D): K is jagged, M and N are fixed per group
				run_gemm_type<typename Configs::GradWeight>(problem, run_with_config);
			} else if (a_row_major && !b_row_major && N > 0) {
				// Forward (2D x 3D, 3D x 3D)
				run_gemm_type<typename Configs::Forward>(problem, [&](auto config) {
					run_ck_tile_group_gemm<DataType, RowMajor, ColMajor, decltype(config)>(group_args.const_data_ptr(), group_count, stream);
				});
			} else if (a_row_major && b_row_major && N > 0) {
				// dX (2D x 3D, 3D x 3D)
				run_gemm_type<typename Configs::GradInput>(problem, [&](auto config) {
					run_ck_tile_group_gemm<DataType, RowMajor, RowMajor, decltype(config)>(group_args.const_data_ptr(), group_count, stream);
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
		// Average tokens per expert, known without reading offs: the jagged dim's total over the groups, else M.
		int64_t avg_tokens = M;
		if (a_is_2d && b_is_2d) {
			group_count = offs->size(0);
			avg_tokens = group_count > 0 ? K / group_count : 0;
			K = -1;
		} else if (a_is_2d) {
			group_count = mat_b.size(0);
			avg_tokens = group_count > 0 ? M / group_count : 0;
			M = -1;
		} else if (b_is_2d) {
			group_count = mat_a.size(0);
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
		auto launch = [&](auto arch_tag, auto dtype_tag) {
			using DataType = decltype(dtype_tag);
			launch_ck_tile_group_gemm<decltype(arch_tag)::value, DataType>(
					mat_a, mat_b, offs, out, group_args, a_row_major, b_row_major,
					static_cast<int32_t>(M), static_cast<int32_t>(N), static_cast<int32_t>(K), static_cast<int32_t>(group_count), avg_tokens);
		};
		auto launch_for_arch = [&](auto dtype_tag) {
			if (at::detail::getCUDAHooks().isGPUArch({"gfx950"})) {
				launch(std::integral_constant<amdgcn_target_id, amdgcn_target_id::GFX950>{}, dtype_tag);
			} else {
				TORCH_CHECK_NOT_IMPLEMENTED(false, "ck_tile group gemm: no tile configs for ", at::cuda::getCurrentDeviceProperties()->gcnArchName);
			}
		};
		switch (mat_a.scalar_type()) {
			case at::kBFloat16:
				launch_for_arch(ck_tile::bf16_t{});
				break;
			case at::kHalf:
				launch_for_arch(ck_tile::half_t{});
				break;
			case at::kFloat:
				TORCH_CHECK(false, "ck_tile group gemm: no fp32 config yet");
				break;
			default:
				TORCH_CHECK(false, "ck_tile group gemm: unsupported dtype ", mat_a.scalar_type());
		}
	}
}
