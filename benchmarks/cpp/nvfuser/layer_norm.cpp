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

static void setupLayerNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const int kReductionAxis = 1;
  const float kEps = 1e-5;

  Double* eps_ptr = new Double(kEps);

  // setup fusion
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);

  fusion->addInput(input);
  fusion->addInput(weight);
  fusion->addInput(bias);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
  }

  auto layer_norm_results = layer_norm(input, 1, weight, bias, eps_ptr);

  auto output = layer_norm_results.output;

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_LayerNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const float kEps = 1e-5;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  std::vector<c10::IValue> aten_inputs({input, weight, bias});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel() + bias.numel()) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void Baseline_LayerNorm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{656, benchmark_state.range(0)};
  const int kReductionAxis = 1;
  std::vector<int64_t> norm_shape;
  for (int idx = kReductionAxis; idx < input_shape.size(); ++idx) {
    norm_shape.push_back(input_shape[idx]);
  }

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);
  at::Tensor bias = at::randn({input_shape[1]}, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::layer_norm(input, norm_shape, weight, bias);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }
}

static void Baseline_LayerNorm_fp32(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Float);
}

static void Baseline_LayerNorm_fp16(benchmark::State& benchmark_state) {
  Baseline_LayerNorm(benchmark_state, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp32_LayerNorm,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp32_LayerNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_fp16_LayerNorm,
    setupLayerNorm,
    NvFuserScheduler_LayerNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_fp16_LayerNorm)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_LayerNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_LayerNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8 << 12}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
