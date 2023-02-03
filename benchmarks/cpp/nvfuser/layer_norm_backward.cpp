#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupLayerNorm_BWD(Fusion* fusion, DataType dtype) {
  FusionGuard fg(fusion);

  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  // setup fusion
  auto grad_out = makeContigTensor(2, dtype);
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);

  auto mean = TensorViewBuilder()
                  .contiguity({false, false})
                  .shape({-1, 1})
                  .dtype(DataType::Float)
                  .build();
  auto rstd = TensorViewBuilder()
                  .contiguity({false, false})
                  .shape({-1, 1})
                  .dtype(DataType::Float)
                  .build();

  fusion->addInput(grad_out);
  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(mean);
  fusion->addInput(rstd);

  if (dtype == DataType::Half) {
    grad_out = castOp(DataType::Float, grad_out);
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto layer_norm_results = layer_norm_backward(
      grad_out, input, {1}, mean, rstd, weight, bias, {true, true, true});

  if (dtype != DataType::Float) {
    layer_norm_results.grad_input =
        castOp(dtype, layer_norm_results.grad_input);
    layer_norm_results.grad_bias = castOp(dtype, layer_norm_results.grad_bias);
    layer_norm_results.grad_weight =
        castOp(dtype, layer_norm_results.grad_weight);
  }

  fusion->addOutput(layer_norm_results.grad_input);
  fusion->addOutput(layer_norm_results.grad_bias);
  fusion->addOutput(layer_norm_results.grad_weight);
}

static void NvFuserScheduler_LayerNorm_BWD(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto maybe_fp16_options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor grad_out = at::randn(input_shape, maybe_fp16_options);
  at::Tensor input = at::randn(input_shape, maybe_fp16_options);
  at::Tensor weight = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor bias = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor mean = at::randn({input_shape[0], 1}, fp32_options);
  at::Tensor rstd = at::randn({input_shape[0], 1}, fp32_options);

  std::vector<c10::IValue> aten_inputs(
      {grad_out, input, weight, bias, mean, rstd});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (3 * input.numel() + weight.numel() + bias.numel() + mean.numel() +
       rstd.numel()) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void Baseline_LayerNorm_BWD(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};
  const size_t kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (auto idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto maybe_fp16_options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor grad_out = at::randn(input_shape, maybe_fp16_options);
  at::Tensor input = at::randn(input_shape, maybe_fp16_options);
  at::Tensor weight = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor bias = at::randn({input_shape[1]}, maybe_fp16_options);
  at::Tensor mean = at::randn({input_shape[0], 1}, fp32_options);
  at::Tensor rstd = at::randn({input_shape[0], 1}, fp32_options);
  std::array<bool, 3> output_mask = {true, true, true};

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    at::native_layer_norm_backward(
        grad_out, input, norm_shape, mean, rstd, weight, bias, output_mask);

    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (3 * input.numel() + weight.numel() + bias.numel() + mean.numel() +
       rstd.numel()) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_LayerNorm_BWD_fp32(benchmark::State& benchmark_state) {
  Baseline_LayerNorm_BWD(benchmark_state, DataType::Float);
}

static void Baseline_LayerNorm_BWD_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNorm_BWD(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_fp32,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 16 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 16 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_LayerNorm_BWD_fp16,
    setupLayerNorm_BWD,
    NvFuserScheduler_LayerNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 16 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 16 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{160, 320}, {2, 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
