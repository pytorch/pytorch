#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <torch/csrc/distributed/c10d/cuda/AsyncMM.cuh>

#include <limits>

// ROCm-only native async-TP GEMM. The CUDA (CUTLASS) implementation lives in
// cuda/AsyncMM.cu; this file holds the AMD Composable Kernel (CK) tile
// implementation and is compiled only on ROCm builds. Keeping the two backends
// in separate files avoids interleaving two entirely different kernel stacks
// behind preprocessor branches.
#if defined(USE_ROCM_ASYNC_MM)

#include <string_view>

#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm/kernel/gemm_kernel.hpp>
#include <ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp>
#include <ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp>
#include <ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp>
#include <ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp>
#include <ck_tile/ops/epilogue.hpp>
#include <ck_tile/ops/gemm.hpp>
#include <ck_tile/core/utility/persistent_async_input_scheduler.hpp>
#include <ck_tile/host/kernel_launch.hpp>

namespace {

struct AsyncGemmConfig
{
    static constexpr ck_tile::index_t M_Tile = 256;
    static constexpr ck_tile::index_t N_Tile = 256;
    // K_Tile=64 regressed row-major B in the reference tuning sweep.
    static constexpr ck_tile::index_t K_Tile = 32;

    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool DoubleSmemBuffer = true;
    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;

    static constexpr bool TransposeC = false;
    static constexpr bool UseStructuredSparsity = false;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool Preshuffle = false;

    static constexpr ck_tile::index_t TilePartitionerGroupNum = 8;
    static constexpr ck_tile::index_t TilePartitionerM01 = 4;
    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;
};

// The kernel is only instantiated for gfx942/gfx950; other arches in
// PYTORCH_ROCM_ARCH get an empty body instead of compiling the CK GEMM.
template <typename Kernel>
struct AsyncInputMMKernel
{
    static constexpr ck_tile::index_t kBlockSize = Kernel::kBlockSize;

    template <typename KernelArgs>
    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
#if defined(__gfx942__) || defined(__gfx950__)
        Kernel{}(kargs);
#endif
    }
};

// gfx942 and gfx950 have 4 shader engines per XCC.
constexpr int kShaderEnginesPerXcc = 4;

void async_input_mm_impl(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_chunk_signals,
    int64_t a_chunk_pivot,
    at::Tensor& out) {
  c10::cuda::CUDAGuard device_guard(a.device());

  const std::string_view arch = at::cuda::getCurrentDeviceProperties()->gcnArchName;
  TORCH_CHECK(
      arch.starts_with("gfx942") || arch.starts_with("gfx950"),
      "async_input_mm: only gfx942 and gfx950 are supported, got ", arch);

  using ElementA = ck_tile::bf16_t;
  using LayoutA = ck_tile::tensor_layout::gemm::RowMajor;
  using ElementB = ck_tile::bf16_t;
  using LayoutB = ck_tile::tensor_layout::gemm::RowMajor;
  using ElementC = ck_tile::bf16_t;
  using LayoutC = ck_tile::tensor_layout::gemm::RowMajor;
  using ElementAccumulator = float;

  using GemmConfig = AsyncGemmConfig;

  using GemmShape = ck_tile::TileGemmShape<
      ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
      ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
      ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
      GemmConfig::PermuteA,
      GemmConfig::PermuteB>;

  using TilePartitioner =
      ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                 GemmConfig::TilePartitionerGroupNum,
                                                 GemmConfig::TilePartitionerM01>;

  using GemmUniversalTraits =
      ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                       GemmConfig::kPadN,
                                       GemmConfig::kPadK,
                                       GemmConfig::DoubleSmemBuffer,
                                       LayoutA,
                                       LayoutB,
                                       LayoutC,
                                       GemmConfig::TransposeC,
                                       GemmConfig::UseStructuredSparsity,
                                       true, // Persistent = True
                                       GemmConfig::NumWaveGroups,
                                       GemmConfig::Preshuffle>;

  using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ElementA,
                                                                     ElementB,
                                                                     ElementAccumulator,
                                                                     GemmShape,
                                                                     GemmUniversalTraits,
                                                                     GemmConfig::Scheduler>;

  using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

  using GemmEpilogue = ck_tile::CShuffleEpilogue<
      ck_tile::CShuffleEpilogueProblem<ElementA,
                                       ElementB,
                                       ck_tile::tuple<>, // No D tensors
                                       ElementAccumulator,
                                       ElementC,
                                       ck_tile::tuple<>, // No D layouts
                                       LayoutC,
                                       ck_tile::element_wise::PassThrough,
                                       TilePartitioner::MPerBlock,
                                       TilePartitioner::NPerBlock,
                                       GemmConfig::M_Warp,
                                       GemmConfig::N_Warp,
                                       GemmConfig::M_Warp_Tile,
                                       GemmConfig::N_Warp_Tile,
                                       GemmConfig::K_Warp_Tile,
                                       UniversalGemmProblem::TransposeC,
                                       GemmConfig::NumWaveGroups,
                                       false, /*FixedVectorSize_*/
                                       1,     /*VectorSizeC_*/
                                       1,     /*BlockedXDLN_PerWarp_*/
                                       GemmConfig::DoubleSmemBuffer>>;

  using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
  static_assert(
      Kernel::UniversalGemmKernel::PersistentKernel,
      "async_input_mm: CK kernel must be persistent");

  const ck_tile::index_t M = a.size(0);
  const ck_tile::index_t N = b.size(1);
  const ck_tile::index_t K = a.size(1);
  const ck_tile::index_t num_chunks = a_chunk_signals.numel();
  const ck_tile::index_t chunk_size_M = M / num_chunks;
  TORCH_CHECK(
      chunk_size_M % GemmConfig::M_Tile == 0,
      "async_input_mm: `a.shape(0) / a_chunk_signals.numel()` must be a multiple of ",
      GemmConfig::M_Tile);
  const ck_tile::index_t tiles_per_chunk = chunk_size_M / GemmConfig::M_Tile;

  ck_tile::PersistentAsyncInputScheduler async_scheduler;
  async_scheduler.tiles_per_chunk_m = tiles_per_chunk;
  async_scheduler.chunk_signals = a_chunk_signals.data_ptr<uint32_t>();
  async_scheduler.tile_idx_pivot_m = a_chunk_pivot * tiles_per_chunk;
  async_scheduler.num_chunks = num_chunks;

  ck_tile::UniversalGemmHostArgs<1, 1, 0> host_args(
      {a.const_data_ptr()},
      {b.const_data_ptr()},
      {},
      out.data_ptr(),
      1, // k_batch
      M,
      N,
      K,
      {K},
      {N},
      {},
      N,
      async_scheduler);
  auto kargs = Kernel::UniversalGemmKernel::MakeKernelArgs(host_args);
  TORCH_CHECK(
      Kernel::UniversalGemmKernel::IsSupportedArgument(kargs),
      "async_input_mm: arguments not supported by the CK tile kernel");

  using LaunchedKernel = AsyncInputMMKernel<Kernel>;
  const dim3 blocks = Kernel::BlockSize();
  const auto kernel_entry = ck_tile::kentry<1, LaunchedKernel, decltype(kargs)>;
  int occupancy = 0;
  C10_CUDA_CHECK(
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel_entry, blocks.x, 0));
  int num_xcc = 8;
#if ROCM_VERSION >= 70000
  C10_CUDA_CHECK(hipDeviceGetAttribute(&num_xcc, hipDeviceAttributeNumberOfXccs, a.get_device()));
#endif
  // The host path copies chunks in pieces of at most 1 MiB, which HIP runs as
  // shader kernels spread over every shader engine. Keep one CU per shader
  // engine out of the persistent grid, or blocks spinning on unsignaled chunks
  // can starve those copies and deadlock.
  const int gemm_cus = std::max(
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount -
          num_xcc * kShaderEnginesPerXcc,
      1);
  const int max_grid = gemm_cus * std::max(occupancy, 1);
  const dim3 grids(std::min<int>(TilePartitioner::GridSize(M, N), max_grid));

  ck_tile::stream_config stream_cfg{at::cuda::getCurrentCUDAStream(), false, 0};
  ck_tile::launch_kernel(
      stream_cfg, ck_tile::make_kernel<1>(LaunchedKernel{}, grids, blocks, 0, kargs));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

#endif // defined(USE_ROCM_ASYNC_MM)

namespace c10d::cuda::detail {

at::Tensor async_input_mm_out(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t a_chunk_pivot,
    at::Tensor out) {
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
      "async_input_mm: `a`, `b` and `out` must be matrices")
  TORCH_CHECK(
      a.is_contiguous() && out.is_contiguous(),
      "async_input_mm: `a` and `out` must be in row-major layout");

  if (!b.is_contiguous()) {
    TORCH_CHECK(b.stride(1) == b.size(0));
    TORCH_CHECK(b.stride(0) == 1);
  }
  TORCH_CHECK_EQ(a.scalar_type(), at::kBFloat16);
  TORCH_CHECK_EQ(b.scalar_type(), at::kBFloat16);
  TORCH_CHECK_EQ(out.scalar_type(), at::kBFloat16);

  int64_t M = a.sizes()[0];
  int64_t N = b.sizes()[1];
  int64_t K = a.sizes()[1];
  TORCH_CHECK_EQ(b.sizes()[0], K);
  TORCH_CHECK_EQ(out.sizes()[0], M);
  TORCH_CHECK_EQ(out.sizes()[1], N);
  // CK indexes tensors with 32-bit integers.
  constexpr int64_t kMaxNumel = std::numeric_limits<int32_t>::max();
  TORCH_CHECK(
      a.numel() <= kMaxNumel && b.numel() <= kMaxNumel && out.numel() <= kMaxNumel,
      "async_input_mm: `a`, `b` and `out` must each have at most ", kMaxNumel, " elements");

  TORCH_CHECK(
      a_chunk_signals.dim() == 1 && a_chunk_signals.numel() > 0 &&
          a_chunk_signals.is_contiguous() &&
          a_chunk_signals.scalar_type() == at::kUInt32 &&
          a_chunk_signals.device() == a.device(),
      "async_input_mm: `a_chunk_signals` must be a non-empty contiguous 1D uint32 tensor on the device of `a`");
  const int64_t num_chunks = a_chunk_signals.numel();
  TORCH_CHECK(
      M % num_chunks == 0,
      "async_input_mm: `a.shape(0)` must be an integer multiple of `a_chunk_signals.numel()`");
  TORCH_CHECK(
      a_chunk_pivot >= 0 && a_chunk_pivot < num_chunks,
      "async_input_mm: `a_chunk_pivot` must be in [0, ", num_chunks, "), got ", a_chunk_pivot);

#if defined(USE_ROCM_ASYNC_MM)
  // The CK kernel is 1.6x (gfx942) to 6x (gfx950) slower with column-major B
  // than with row-major B, which outweighs copying B to row-major.
  async_input_mm_impl(a, b.contiguous(), a_chunk_signals, a_chunk_pivot, out);
#else
  TORCH_CHECK(
      false,
      "async_input_mm: PyTorch was built without ROCm gfx942/gfx950 support for this kernel");
#endif
  return out;
}

at::Tensor async_input_mm(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t a_chunk_pivot) {
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2,
      "async_input_mm: `a`, `b` and `out` must all be a matrix")

  int64_t M = a.sizes()[0];
  int64_t N = b.sizes()[1];
  auto out = a.new_empty({M, N});
  return async_input_mm_out(a, b, a_chunk_signals, a_chunk_pivot, out);
}

} // namespace c10d::cuda::detail
