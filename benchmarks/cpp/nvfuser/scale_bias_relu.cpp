#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static void setupFusion(
    Fusion* fusion,
    const size_t kNumberOfDims,
    TensorView* x_half,
    TensorView* scale_half,
    TensorView* bias_half) {
  FusionGuard fg(fusion);

  fusion->addInput(x_half);
  fusion->addInput(scale_half);
  fusion->addInput(bias_half);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  for (size_t axis = 0; axis < kNumberOfDims - 1; ++axis) {
    broadcast_mask[axis] = true;
  }

  auto x = castOp(DataType::Float, x_half);
  auto scale = castOp(DataType::Float, scale_half);
  auto bias = castOp(DataType::Float, bias_half);

  auto scale_bias = add(mul(x, scale), bias);
  auto scale_bias_relu = unaryOp(UnaryOpType::Relu, scale_bias);

  auto scale_bias_relu_half = castOp(DataType::Half, scale_bias_relu);

  fusion->addOutput(scale_bias_relu_half);
}

static void setupFusion(
    Fusion* fusion,
    const size_t kNumberOfDims,
    TensorView* x_half,
    TensorView* weight_half,
    TensorView* bias_half,
    TensorView* mean_half,
    TensorView* var_half) {
  FusionGuard fg(fusion);

  fusion->addInput(x_half);
  fusion->addInput(weight_half);
  fusion->addInput(bias_half);
  fusion->addInput(mean_half);
  fusion->addInput(var_half);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  for (size_t axis = 0; axis < kNumberOfDims - 1; ++axis) {
    broadcast_mask[axis] = true;
  }

  auto x = castOp(DataType::Float, x_half);
  auto weight = castOp(DataType::Float, weight_half);
  auto bias = castOp(DataType::Float, bias_half);
  auto mean = castOp(DataType::Float, mean_half);
  auto var = castOp(DataType::Float, var_half);

  auto rsqrt = unaryOp(UnaryOpType::Rsqrt, var);
  auto this_scale = mul(weight, rsqrt);
  auto this_bias = mul(sub(bias, mean), this_scale);

  auto bcast_scale = broadcast(this_scale, broadcast_mask);
  auto bcast_bias = broadcast(this_bias, broadcast_mask);

  auto scale_bias = add(mul(x, bcast_scale), bcast_bias);
  auto scale_bias_relu = unaryOp(UnaryOpType::Relu, scale_bias);

  auto scale_bias_relu_half = castOp(DataType::Half, scale_bias_relu);

  fusion->addOutput(scale_bias_relu_half);
}

//------------------------------------------------------------------------------

static void SBR_NvFuser_Multiple(benchmark::State& benchmark_state) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{1, 1, 1, -1};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = TensorViewBuilder()
               .ndims(input_shape.size())
               .dtype(DataType::Half)
               .build();
  auto scale =
      TensorViewBuilder().shape(bcast_shape).dtype(DataType::Half).build();
  auto bias =
      TensorViewBuilder().shape(bcast_shape).dtype(DataType::Half).build();

  // setup fusion
  setupFusion(&fusion, input_shape.size(), x, scale, bias);

  // inputs
  at::manual_seed(0);
  std::vector<int64_t> static_bcast_shape{1, 1, 1, benchmark_state.range(2)};
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_scale = at::ones(static_bcast_shape, options);
  at::Tensor at_bias = at::zeros(static_bcast_shape, options);

  // inputs
  std::vector<c10::IValue> inputs = {at_x, at_scale, at_bias};

  // outputs
  std::vector<at::Tensor> outputs;

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(DataType::Half)));
}

static void SBR_Baseline_Multiple(benchmark::State& benchmark_state) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_scale = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto scale = at::mul(at_x, at_scale);
    auto bias = at::add(scale, at_bias);
    auto output = at::relu(bias);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(DataType::Half)));
}

//------------------------------------------------------------------------------

static void SBR_NvFuser(benchmark::State& benchmark_state) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{benchmark_state.range(2)};

  Fusion fusion;
  FusionGuard fg(&fusion);

  auto x = TensorViewBuilder()
               .ndims(input_shape.size())
               .dtype(DataType::Half)
               .build();
  auto weight = TensorViewBuilder()
                    .ndims(bcast_shape.size())
                    .dtype(DataType::Half)
                    .build();
  auto bias = TensorViewBuilder()
                  .ndims(bcast_shape.size())
                  .dtype(DataType::Half)
                  .build();
  auto mean = TensorViewBuilder()
                  .ndims(bcast_shape.size())
                  .dtype(DataType::Half)
                  .build();
  auto var = TensorViewBuilder()
                 .ndims(bcast_shape.size())
                 .dtype(DataType::Half)
                 .build();

  // setup fusion
  setupFusion(&fusion, input_shape.size(), x, weight, bias, mean, var);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);
  at::Tensor at_mean = at::zeros(bcast_shape, options);
  at::Tensor at_var = at::ones(bcast_shape, options);

  // inputs
  std::vector<c10::IValue> inputs = {at_x, at_weight, at_bias, at_mean, at_var};

  // outputs
  std::vector<at::Tensor> outputs;

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  // fusion.printMath();
  // fusion.printKernel();
  // TORCH_INTERNAL_ASSERT(false);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(DataType::Half)));
}

static void SBR_Baseline(benchmark::State& benchmark_state) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{1, 1, 1, benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);
  at::Tensor at_mean = at::zeros(bcast_shape, options);
  at::Tensor at_var = at::ones(bcast_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto this_scale = at::mul(at_weight, at::rsqrt(at_var));
    auto this_bias = at::mul(at::sub(at_bias, at_mean), this_scale);

    auto scale = at::mul(at_x, this_scale);
    auto bias = at::add(scale, this_bias);
    auto output = at::relu(bias);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(DataType::Half)));
}

//------------------------------------------------------------------------------

BENCHMARK(SBR_NvFuser_Multiple)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(SBR_Baseline_Multiple)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(SBR_NvFuser)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(SBR_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
