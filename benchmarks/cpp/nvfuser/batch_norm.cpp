#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include "utils.h"

using namespace torch::jit::fuser::cuda;

static TensorView* setupBatchNorm(
    Fusion* fusion,
    TensorView* input,
    TensorView* weight,
    TensorView* bias,
    const int kNumberOfDims) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  std::vector<int> reduction_axes;
  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (size_t axis = 0; axis < kNumberOfDims; ++axis) {
    if (axis != 1) {
      reduction_axes.push_back(axis);
      broadcast_mask[axis] = true;
      num_features =
          mul(num_features, input->domain()->domain()[axis]->extent());
    }
  }

  auto x_sum = sum(input, reduction_axes);
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  auto x_mean = div(x_sum_bcast, num_features);

  auto x_mean_sub = sub(input, x_mean);
  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  auto var = div(var_sum_bcast, num_features);

  auto var_eps = add(var, new Double(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto norm = mul(x_mean_sub, rvar);

  auto weight_bcast = broadcast(weight, broadcast_mask);
  auto bias_bcast = broadcast(bias, broadcast_mask);
  auto norm_gamma = mul(norm, weight_bcast);
  auto norm_gamma_bias = add(norm_gamma, bias_bcast);
  return norm_gamma_bias;
}

//------------------------------------------------------------------------------

static void MagicScheduler_BatchNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  auto weight = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  auto bias = TensorViewBuilder().ndims(1).dtype(DataType::Float).build();
  fusion.addInput(input);
  fusion.addInput(weight);
  fusion.addInput(bias);

  auto output =
      setupBatchNorm(&fusion, input, weight, bias, input_shape.size());
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  std::vector<c10::IValue> inputs({at_x, at_weight, at_bias});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params =
      getNormalizationHeuristics(&fusion, inputs, reduction_tensors);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleNormalization(
      &fusion, reduction_params.value(), reduction_tensors, other_tensors);

  FusionExecutor executor;
  executor.setMeasureKernelTimeFlag(true);
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(
        c10::ArrayRef<c10::IValue>(inputs), reduction_params.value().lparams);
    benchmark_state.SetIterationTime(executor.kernelTimeMs() / 1000.0);
    cudaDeviceSynchronize();
  }
}

static void MagicScheduler_BatchNorm_Baseline(
    benchmark::State& benchmark_state) {
  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      32,
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones({input_shape[1]}, options);
  at::Tensor at_bias = at::zeros({input_shape[1]}, options);
  at::Tensor at_mean = at::zeros({input_shape[1]}, options);
  at::Tensor at_var = at::ones({input_shape[1]}, options);

  auto ato_weight = c10::optional<at::Tensor>(at_weight);
  auto ato_bias = c10::optional<at::Tensor>(at_bias);
  auto ato_running_mean = c10::optional<at::Tensor>(at_mean);
  auto ato_running_var = c10::optional<at::Tensor>(at_var);

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
}

BENCHMARK(MagicScheduler_BatchNorm)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_BatchNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {8, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
