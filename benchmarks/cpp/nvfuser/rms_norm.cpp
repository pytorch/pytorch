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

static void setupRMSNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  FusionGuard fg(fusion);

  const float kEps = 1e-6;

  Double* eps_ptr = IrBuilder::create<Double>(kEps);

  // setup fusion
  auto input = makeContigTensor(3, dtype);
  auto weight = makeContigTensor(1, dtype);

  fusion->addInput(input);
  fusion->addInput(weight);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
  }

  auto rms_norm_results = rms_norm(input, 1, weight, eps_ptr);

  auto output = rms_norm_results.output;

  if (dtype != DataType::Float) {
    output = castOp(dtype, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_RMSNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  std::vector<int64_t> input_shape{8, benchmark_state.range(0), 1024};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[2]}, options);

  std::vector<c10::IValue> aten_inputs({input, weight});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel()) * int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_fp32,
    setupRMSNorm,
    NvFuserScheduler_RMSNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{16, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{18, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{22, 44}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{24, 48}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_fp16,
    setupRMSNorm,
    NvFuserScheduler_RMSNorm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{16, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{18, 56}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{22, 44}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{24, 48}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TODO: Automatically disable/enable if bf16 is supported
// NVFUSER_BENCHMARK_DEFINE(
//     NvFuserScheduler_RMSNorm_bf16,
//     setupRMSNorm,
//     NvFuserScheduler_RMSNorm,
//     DataType::BFloat16);

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{16, 64}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{18, 56}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{22, 44}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{24, 48}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();
