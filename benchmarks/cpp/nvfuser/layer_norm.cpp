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

static void MagicScheduler_LayerNorm(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  const float kEps = 1e-5;

  std::vector<int64_t> norm_shape;
  for (int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }
  Double* eps_ptr = new Double(kEps);

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);
  auto layer_norm_results =
      layer_norm(input, norm_shape, nullptr, nullptr, eps_ptr);
  fusion.addOutput(layer_norm_results.output);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs({at_x});

  // outputs
  std::vector<at::Tensor> outputs;

  auto reduction_params = getNormalizationHeuristics(&fusion, inputs);
  TORCH_CHECK(reduction_params, "Reduction schedule was not generated!");

  scheduleNormalization(&fusion, reduction_params.value());

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
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_LayerNorm_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
