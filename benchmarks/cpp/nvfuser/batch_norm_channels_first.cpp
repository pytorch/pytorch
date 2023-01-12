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

static void setupBatchNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto momentum_ptr = IrBuilder::create<Double>(kMomentum);
  auto eps_ptr = IrBuilder::create<Double>(kEps);

  auto result = batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr);

  auto output = result.output;

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_BatchNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_run_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_run_var = at::ones({input_shape[1]}, fp32_options);
  std::vector<c10::IValue> aten_inputs(
      {at_x, at_weight, at_bias, at_run_mean, at_run_var});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      ((2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
           int64_t(dataTypeSize(dtype)) +
       (2 * (at_run_mean.numel() + at_run_var.numel()) *
        int64_t(dataTypeSize(DataType::Float)))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_run_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_run_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_run_mean = c10::optional<at::Tensor>(at_run_mean);
  auto ato_run_var = c10::optional<at::Tensor>(at_run_var);

  auto output = at::batch_norm(
      at_x,
      ato_weight,
      ato_bias,
      ato_run_mean,
      ato_run_var,
      true,
      kMomentum,
      kEps,
      true);

  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::batch_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_run_mean,
        ato_run_var,
        true,
        kMomentum,
        kEps,
        true);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      ((2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
           int64_t(dataTypeSize(dtype)) +
       (2 * (at_run_mean.numel() + at_run_var.numel()) *
        int64_t(dataTypeSize(DataType::Float)))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_cuDNN_fp32(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Float);
}

static void Baseline_BatchNorm_cuDNN_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Half);
}

// Simple aliases just for names in the printed output
static void Baseline_ResNet_BatchNorm_cuDNN_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Half);
}

static void Baseline_ResNext_BatchNorm_cuDNN_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_fp32,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_fp16,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_BatchNorm_cuDNN_fp32)
    // ->RangeMultiplier(2)
    // cuDNN didn't make it to 1024
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_cuDNN_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
// RESNET and REXNEXT benchmarks

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNet_BatchNorm_fp16,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNet_BatchNorm_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNext_BatchNorm_fp16,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNext_BatchNorm_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_ResNet_BatchNorm_cuDNN_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_ResNext_BatchNorm_cuDNN_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
