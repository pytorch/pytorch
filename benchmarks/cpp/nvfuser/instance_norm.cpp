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

static void setupFusionHalf(
    Fusion* fusion,
    const size_t kNumberOfDims,
    TensorView* x_half,
    TensorView* weight_half,
    TensorView* bias_half,
    TensorView* mean,
    TensorView* var) {
  FusionGuard fg(fusion);

  fusion->addInput(x_half);
  fusion->addInput(weight_half);
  fusion->addInput(bias_half);
  fusion->addInput(mean);
  fusion->addInput(var);

  auto x = castOp(DataType::Float, x_half);
  auto weight = castOp(DataType::Float, weight_half);
  auto bias = castOp(DataType::Float, bias_half);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  auto momentum_ptr = new Double(kMomentum);
  auto eps_ptr = new Double(kEps);

  auto norm = instance_norm(
      x, weight, bias, mean, var, kTraining, momentum_ptr, eps_ptr);
  auto norm_relu = unaryOp(UnaryOpType::Relu, norm.output);

  auto norm_relu_half = castOp(DataType::Half, norm_relu);

  fusion->addOutput(norm_relu_half);
}

static void setupFusionFloat(
    Fusion* fusion,
    const size_t kNumberOfDims,
    TensorView* x,
    TensorView* weight,
    TensorView* bias,
    TensorView* mean,
    TensorView* var) {
  FusionGuard fg(fusion);

  fusion->addInput(x);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(mean);
  fusion->addInput(var);

  const bool kTraining = true;
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  auto momentum_ptr = new Double(kMomentum);
  auto eps_ptr = new Double(kEps);

  auto norm = instance_norm(
      x, weight, bias, mean, var, kTraining, momentum_ptr, eps_ptr);
  auto norm_relu = unaryOp(UnaryOpType::Relu, norm.output);

  fusion->addOutput(norm_relu);
}

//------------------------------------------------------------------------------

static void InstanceNorm_NvFuser(
    benchmark::State& benchmark_state,
    DataType dtype) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(2),
      benchmark_state.range(1),
      benchmark_state.range(1)};
  const auto aten_dtype = data_type_to_aten(dtype);

  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  Fusion& fusion = *fusion_ptr.get();
  FusionGuard fg(&fusion);

  auto x = TensorViewBuilder().ndims(input_shape.size()).dtype(dtype).build();
  auto weight = TensorViewBuilder().ndims(1).dtype(dtype).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(dtype).build();
  auto running_mean =
      TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto running_var =
      TensorViewBuilder().ndims(1).dtype(DataType::Float).build();

  // setup fusion
  switch (dtype) {
    case DataType::Float: {
      setupFusionFloat(
          &fusion,
          input_shape.size(),
          x,
          weight,
          bias,
          running_mean,
          running_var);
      break;
    }
    case DataType::Half: {
      setupFusionHalf(
          &fusion,
          input_shape.size(),
          x,
          weight,
          bias,
          running_mean,
          running_var);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported DataType.")
      break;
  }

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(aten_dtype).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor at_var = at::ones({input_shape[1]}, fp32_options);

  std::vector<c10::IValue> inputs = {at_x, at_weight, at_bias, at_mean, at_var};
  std::vector<at::Tensor> outputs;

  FusionExecutorCache fec(std::move(fusion_ptr));

  // Run a single iteration first to compile fusion
  // Avoid measuring compile time in benchmark
  fec.runFusionWithInputs(inputs);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    outputs = fec.runFusionWithInputs(inputs);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2) * dataTypeSize(DataType::Float)));
}

static void InstanceNorm_Baseline(
    benchmark::State& benchmark_state,
    DataType dtype) {
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
  }

  const size_t kSize =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t kChannels = input_shape[1];

  // Read: x, weight, bias
  // Write: y, running_mean, running_var
  benchmark_state.SetBytesProcessed(
      benchmark_state.iterations() *
      ((kChannels * 2 + kSize * 2) * dataTypeSize(dtype) +
       (kChannels * 2) * dataTypeSize(DataType::Float)));
}

//------------------------------------------------------------------------------

static void InstanceNorm_NvFuser_fp32(benchmark::State& benchmark_state) {
  InstanceNorm_NvFuser(benchmark_state, DataType::Float);
}

static void InstanceNorm_Baseline_fp32(benchmark::State& benchmark_state) {
  InstanceNorm_Baseline(benchmark_state, DataType::Float);
}

static void InstanceNorm_NvFuser_fp16(benchmark::State& benchmark_state) {
  InstanceNorm_NvFuser(benchmark_state, DataType::Half);
}

static void InstanceNorm_Baseline_fp16(benchmark::State& benchmark_state) {
  InstanceNorm_Baseline(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

BENCHMARK(InstanceNorm_NvFuser_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(InstanceNorm_Baseline_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(InstanceNorm_NvFuser_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(InstanceNorm_Baseline_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
