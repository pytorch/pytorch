#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static void setupInstanceNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

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

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  auto momentum_ptr = new Double(kMomentum);
  auto eps_ptr = new Double(kEps);

  auto norm = instance_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      kTraining,
      momentum_ptr,
      eps_ptr);

  auto output = unaryOp(UnaryOpType::Relu, norm.output);

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_InstanceNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
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
  at::Tensor at_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_var = at::ones({input_shape[1]}, fp32_options);

  std::vector<c10::IValue> aten_inputs = {
      at_x, at_weight, at_bias, at_mean, at_var};
  std::vector<at::Tensor> outputs;

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias, running_mean, running_var
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2 * 2) * dataTypeSize(DataType::Float)));
}

static void Baseline_InstanceNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  const auto aten_dtype = data_type_to_aten(dtype);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto norm = at::instance_norm(
        at_x,
        ato_weight,
        ato_bias,
        ato_running_mean,
        ato_running_var,
        true,
        kMomentum,
        kEps,
        false);
    auto output = at::relu(norm);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }

  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias, running_mean, running_var
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2 * 2) * dataTypeSize(DataType::Float)));
}

//------------------------------------------------------------------------------

static void Baseline_InstanceNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_InstanceNorm(benchmark_state, DataType::Float);
}

static void Baseline_InstanceNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_InstanceNorm(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp32_InstanceNorm,
    setupInstanceNorm,
    NvFuserScheduler_InstanceNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp32_InstanceNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp16_InstanceNorm,
    setupInstanceNorm,
    NvFuserScheduler_InstanceNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp16_InstanceNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
//------------------------------------------------------------------------------

BENCHMARK(Baseline_InstanceNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_InstanceNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
