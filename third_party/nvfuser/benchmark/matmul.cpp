#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/matmul.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

bool cudaArchGuardShouldSkip(int required_major, int required_minor) {
  int capability_major = at::cuda::getCurrentDeviceProperties()->major;
  int capability_minor = at::cuda::getCurrentDeviceProperties()->minor;

  if (capability_major < required_major ||
      (capability_major == required_major &&
       capability_minor < required_minor)) {
    return true;
  }
  return false;
}

bool hasRequiredSmemSize(size_t required_size) {
  // Only checking device 0
  return at::cuda::getDeviceProperties(0)->sharedMemPerBlockOptin >=
      required_size;
}

#define NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(                       \
    REQUIRED_MAJOR, REQUIRED_MINOR, SMEM_SIZE, STATE)            \
  if (cudaArchGuardShouldSkip(REQUIRED_MAJOR, REQUIRED_MINOR) || \
      !hasRequiredSmemSize(SMEM_SIZE)) {                         \
    STATE.SkipWithError("Unsupported arch or not enough smem!"); \
    return;                                                      \
  }

// util to track support matmul operand layout.
using MatmulLayout = MmaOptions::MmaInputLayout;

static constexpr std::array<MatmulLayout, 3> kAllSupportedLayout = {
    MatmulLayout::TT,
    MatmulLayout::NT,
    MatmulLayout::TN};

// Generic interface to get matmul op with the given layout.
TensorView* matmul(TensorView* a, TensorView* b, MatmulLayout layout) {
  TORCH_CHECK(
      a->nDims() == 2 && b->nDims() == 2, "only pure matmuls for these tests");
  TensorView *tv2 = nullptr, *tv0b = nullptr, *tv1b = nullptr;
  switch (layout) {
    case MatmulLayout::TT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {1});
      break;
    case MatmulLayout::TN:
      tv0b = broadcast(a, {false, true, false});
      tv1b = broadcast(b, {true, false, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {2});
      break;
    case MatmulLayout::NT:
      tv0b = broadcast(a, {false, false, true});
      tv1b = broadcast(b, {false, true, false});
      tv2 = fusedMultiplySum(tv0b, tv1b, {0});
      break;
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return tv2;
}

// Utility to generate matmul input tensors based on given layout
at::Tensor atMatmul(at::Tensor a, at::Tensor b, MatmulLayout layout) {
  switch (layout) {
    case MatmulLayout::TT:
      return a.matmul(b);
    case MatmulLayout::TN:
      return a.matmul(b.t());
    case MatmulLayout::NT:
      return a.t().matmul(b);
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return at::Tensor();
}

// Utility to generate reference results based on given layout
std::pair<at::Tensor, at::Tensor> fp16MatmulAtInput(
    int M,
    int N,
    int K,
    MatmulLayout layout) {
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);

  switch (layout) {
    case MatmulLayout::TT:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({K, N}, options));
    case MatmulLayout::TN:
      return std::make_pair(
          at::randn({M, K}, options), at::randn({N, K}, options));
    case MatmulLayout::NT:
      return std::make_pair(
          at::randn({K, M}, options), at::randn({K, N}, options));
    default:
      TORCH_CHECK(false, "unsupported data layout.");
  }
  return std::make_pair(at::Tensor(), at::Tensor());
}

// TODO: separate compute and schedule definition once the can schedule
//  logic and pattern matching is ready.
void setupMatmul(Fusion* fusion, MatmulLayout layout, MatmulParam params) {
  // Only hgemm on the initial setup
  auto a = makeContigTensor(2, DataType::Half);
  auto b = makeContigTensor(2, DataType::Half);

  auto c = matmul(a, b, layout);

  fusion->addInput(a);
  fusion->addInput(b);
  fusion->addOutput(c);

  scheduleMatmul(c, a, b, params);
}

static void SingleMatmulBase(
    benchmark::State& benchmark_state,
    MatmulLayout layout,
    MatmulParam params) {
  std::vector<int64_t> input_mnk{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // Define fusion graph
  setupMatmul(fusion, layout, params);

  // inputs
  at::manual_seed(0);

  // Tensor inputs
  auto inputs = fp16MatmulAtInput(
      input_mnk.at(0), input_mnk.at(1), input_mnk.at(2), layout);

  KernelArgumentHolder args = KernelArgumentHolder::createKernelArgumentHolder(
      {inputs.first, inputs.second});

  // Always use 32b indexing mode for now.
  TORCH_INTERNAL_ASSERT(args.getIndexMode() == KernelIndexMode::INT32);

  // Compile kernel
  FusionExecutor fe;
  fe.compileFusion(fusion, args, LaunchParams());

  // Warm up run
  auto outputs = fe.runFusion({inputs.first, inputs.second});
  fe.setMeasureKernelTimeFlag(true);

  // Sync everything up before we start
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto outputs = fe.runFusion({inputs.first, inputs.second});
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();

  // TODO: FLOPS calculation
}

static void EagerModeMatmul(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  std::vector<int64_t> input_mnk{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2)};

  at::manual_seed(0);

  auto inputs = fp16MatmulAtInput(
      input_mnk.at(0), input_mnk.at(1), input_mnk.at(2), layout);

  // warm up run
  auto outputs = atMatmul(inputs.first, inputs.second, layout);

  for (auto _ : benchmark_state) {
    clearL2Cache();
    CudaKernelTimer timer;
    outputs = atMatmul(inputs.first, inputs.second, layout);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();
}

// Actual benchmarking
// -----------------------------------------------------------------

size_t getSmemSize(GemmTile cta_tile, int stage_number) {
  return ((cta_tile.m * cta_tile.k) + (cta_tile.n * cta_tile.k)) *
      dataTypeSize(DataType::Half) * stage_number;
}

// TODO: this part eventually will be automated by heuristics
MatmulParam getMatmulParams(
    GemmTile cta_tile,
    int stage_number,
    MatmulLayout layout) {
  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = cta_tile;
  // TODO: pipe through split K
  gemm_tile.warp_tile = GemmTile(64, 64, cta_tile.k);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  // Collect mma swizzle info
  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_16_16, gemm_tile)
          .layout(layout);

  MatmulParam params(mma_builder);
  params.tile_sizes = gemm_tile;
  params.async_gmem_load_operands = true;
  params.double_buffer_options.double_buffer_smem_write = true;
  params.double_buffer_options.double_buffer_smem_read = true;
  params.double_buffer_options.smem_double_buffer_stage = stage_number;

  return params;
}

static void Nvfuser_Matmul_4warp3stage(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  auto cta_tile = GemmTile(128, 128, 32);
  int number_of_stage = 3;

  auto params = getMatmulParams(cta_tile, number_of_stage, layout);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  SingleMatmulBase(benchmark_state, layout, params);
}

static void Nvfuser_Matmul_8warp3stage(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  auto cta_tile = GemmTile(256, 128, 32);
  int number_of_stage = 3;

  auto params = getMatmulParams(cta_tile, number_of_stage, layout);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  SingleMatmulBase(benchmark_state, layout, params);
}

static void Nvfuser_Matmul_4warp4stage(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  auto cta_tile = GemmTile(128, 128, 32);
  int number_of_stage = 4;

  auto params = getMatmulParams(cta_tile, number_of_stage, layout);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  SingleMatmulBase(benchmark_state, layout, params);
}

static void Nvfuser_Matmul_8warp4stage(
    benchmark::State& benchmark_state,
    MatmulLayout layout) {
  auto cta_tile = GemmTile(256, 128, 32);
  int number_of_stage = 4;

  auto params = getMatmulParams(cta_tile, number_of_stage, layout);

  NVFUSER_BENCHMARK_ARCH_SMEM_GUARD(
      8, 0, getSmemSize(cta_tile, number_of_stage), benchmark_state);

  // Run benchmark:
  SingleMatmulBase(benchmark_state, layout, params);
}

// ----------------------------- Benchmark Instantiation-------

// Common utils:
#define NO_TILE_QUANTIZATION_ARGS                                             \
  ArgsProduct(                                                                \
      {{2048}, {3456}, benchmark::CreateDenseRange(512, 4096, /*step=*/512)}) \
      ->Unit(benchmark::kMicrosecond)                                         \
      ->UseManualTime();

#define ForAllLayouts(run)   \
  run(TT, MatmulLayout::TT); \
  run(TN, MatmulLayout::TN); \
  run(NT, MatmulLayout::NT)

// Instantiations:
#define Nvfuser_4warp3stage_test(layout_label, layout) \
  BENCHMARK_CAPTURE(                                   \
      Nvfuser_Matmul_4warp3stage,                      \
      no_quant_nvfuser_4warp_##layout_label,           \
      layout)                                          \
      ->NO_TILE_QUANTIZATION_ARGS

#define Nvfuser_8warp3stage_test(layout_label, layout) \
  BENCHMARK_CAPTURE(                                   \
      Nvfuser_Matmul_8warp3stage,                      \
      no_quant_nvfuser_8warp_##layout_label,           \
      layout)                                          \
      ->NO_TILE_QUANTIZATION_ARGS

#define Nvfuser_4warp4stage_test(layout_label, layout) \
  BENCHMARK_CAPTURE(                                   \
      Nvfuser_Matmul_4warp4stage,                      \
      no_quant_nvfuser_4warp_##layout_label,           \
      layout)                                          \
      ->NO_TILE_QUANTIZATION_ARGS

#define Nvfuser_8warp4stage_test(layout_label, layout) \
  BENCHMARK_CAPTURE(                                   \
      Nvfuser_Matmul_8warp4stage,                      \
      no_quant_nvfuser_8warp_##layout_label,           \
      layout)                                          \
      ->NO_TILE_QUANTIZATION_ARGS

#define Eagermode_test(layout_label, layout)                      \
  BENCHMARK_CAPTURE(                                              \
      EagerModeMatmul, no_quant_eagermode_##layout_label, layout) \
      ->NO_TILE_QUANTIZATION_ARGS

ForAllLayouts(Nvfuser_4warp3stage_test);
ForAllLayouts(Nvfuser_4warp4stage_test);
ForAllLayouts(Nvfuser_8warp3stage_test);
ForAllLayouts(Nvfuser_8warp4stage_test);
ForAllLayouts(Eagermode_test);
