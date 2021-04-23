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

static TensorView* setupLayerNorm(
    Fusion* fusion,
    TensorView* input,
    const int kNumberOfDims,
    std::vector<int64_t>& norm_shape) {
  FusionGuard fg(fusion);

  const float kEps = 1e-5;
  std::vector<int> reduction_axes(norm_shape.size());
  std::vector<bool> broadcast_mask(input->nDims(), false);
  torch::jit::fuser::cuda::Val* num_features = new Double(1);
  for (int idx = 0; idx < norm_shape.size(); ++idx) {
    const int axis = input->nDims() - 1 - idx;
    reduction_axes[idx] = axis;
    broadcast_mask[axis] = true;
    num_features = mul(num_features, input->domain()->domain()[axis]->extent());
  }

  // Reduction
  auto x_sum = sum(input, reduction_axes);
  // Broadcast
  auto x_sum_bcast = broadcast(x_sum, broadcast_mask);
  // Point-wise
  auto x_mean = div(x_sum_bcast, num_features);
  auto x_mean_sub = sub(input, x_mean);

  auto x_mean_sub_pow = mul(x_mean_sub, x_mean_sub);
  // Reduction
  auto var_sum = sum(x_mean_sub_pow, reduction_axes);
  // Broadcast
  auto var_sum_bcast = broadcast(var_sum, broadcast_mask);
  // Point-wise
  auto var = div(var_sum_bcast, num_features);
  auto var_eps = add(var, new Double(kEps));
  auto rvar = unaryOp(UnaryOpType::Rsqrt, var_eps);
  auto output = mul(x_mean_sub, rvar);
  return output;
}

//------------------------------------------------------------------------------

static void MagicScheduler_LayerNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);
  auto output = setupLayerNorm(&fusion, input, input_shape.size(), norm_shape);
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

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

static void MagicScheduler_LayerNorm_Baseline(
    benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(at_x, norm_shape);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_LayerNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 13}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_LayerNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 13}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
