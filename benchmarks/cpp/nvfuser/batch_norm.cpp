#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupBatchNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(4)
                   .dtype(dtype)
                   .contiguity(std::vector<bool>(4, true))
                   .build();
  auto weight = TensorViewBuilder()
                    .ndims(1)
                    .dtype(dtype)
                    .contiguity(std::vector<bool>(1, true))
                    .build();
  auto bias = TensorViewBuilder()
                  .ndims(1)
                  .dtype(dtype)
                  .contiguity(std::vector<bool>(1, true))
                  .build();
  auto running_mean = TensorViewBuilder()
                          .ndims(1)
                          .dtype(DataType::Float)
                          .contiguity(std::vector<bool>(1, true))
                          .build();
  auto running_var = TensorViewBuilder()
                         .ndims(1)
                         .dtype(DataType::Float)
                         .contiguity(std::vector<bool>(1, true))
                         .build();
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

  auto momentum_ptr = new Double(kMomentum);
  auto eps_ptr = new Double(kEps);

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

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

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
      (int64_t(benchmark_state.iterations()) *
       (2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
       int64_t(dataTypeSize(dtype))) +
      (2 * (at_run_mean.numel() + at_run_var.numel()) *
       int64_t(dataTypeSize(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_running_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_running_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_running_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_running_var);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::batch_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
  benchmark_state.SetBytesProcessed(
      (int64_t(benchmark_state.iterations()) *
       (2 * (at_x.numel() + at_weight.numel() + at_bias.numel())) *
       int64_t(dataTypeSize(dtype))) +
      (2 * (at_running_mean.numel() + at_running_var.numel()) *
       int64_t(dataTypeSize(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Float);
}

static void Baseline_BatchNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp32_BatchNorm,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp32_BatchNorm)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp16_BatchNorm,
    setupBatchNorm,
    NvFuserScheduler_BatchNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp16_BatchNorm)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_BatchNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 32}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
