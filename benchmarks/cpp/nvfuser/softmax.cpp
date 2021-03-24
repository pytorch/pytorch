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

static TensorView* setupSoftmax(
    Fusion* fusion,
    TensorView* input,
    const int kNumberOfDims,
    const int kReductionAxis) {
  FusionGuard fg(fusion);

  std::vector<bool> broadcast_mask(kNumberOfDims, false);
  broadcast_mask[kReductionAxis] = true;

  auto max_val = max(input, {kReductionAxis});
  auto bcast_max = broadcast(max_val, broadcast_mask);
  auto x_max_sub = sub(input, bcast_max);
  auto exp = unaryOp(UnaryOpType::Exp, x_max_sub);
  auto sum_exp = sum(exp, {kReductionAxis});
  auto bcast_sum = broadcast(sum_exp, broadcast_mask);
  auto output = div(exp, bcast_sum);
  return output;
}

//------------------------------------------------------------------------------

static void MagicScheduler_Softmax(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // setup fusion
  auto input = TensorViewBuilder()
                   .ndims(input_shape.size())
                   .dtype(DataType::Float)
                   .build();
  fusion.addInput(input);
  auto output =
      setupSoftmax(&fusion, input, input_shape.size(), kReductionAxis);
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

static void MagicScheduler_Softmax_Baseline(benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{
      benchmark_state.range(1), benchmark_state.range(0)};
  const int kReductionAxis = benchmark_state.range(2);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);

  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::_softmax(at_x, kReductionAxis, false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_Softmax)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 13}, {0, 1}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_Softmax_Baseline)
    ->RangeMultiplier(2)
    ->Ranges({{656, 656}, {8, 8 << 13}, {0, 1}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void MagicScheduler_Softmax_Dropout(benchmark::State& benchmark_state) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};
  const int kReductionAxis = 3;

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;

  // setup fusion
  auto attention_scores = TensorViewBuilder()
                              .ndims(input_shape.size())
                              .dtype(DataType::Float)
                              .build();
  auto attention_mask = TensorViewBuilder()
                            .ndims(input_shape.size())
                            .dtype(DataType::Float)
                            .build();
  Double* divisor = new Double();
  fusion.addInput(attention_scores);
  fusion.addInput(attention_mask);
  fusion.addInput(divisor);

  attention_scores = div(attention_scores, divisor);
  attention_scores = add(attention_scores, attention_mask);
  auto attention_probs = setupSoftmax(
      &fusion, attention_scores, input_shape.size(), kReductionAxis);
  auto random = unaryOp(UnaryOpType::RandLike, attention_probs);
  auto mask =
      binaryOp(BinaryOpType::LT, random, new Double(kDropoutProbability));
  auto float_mask = castOp(DataType::Float, mask);
  auto dropout = mul(attention_probs, float_mask);
  auto output = mul(dropout, new Double(1.0f / kDropoutProbability));

  fusion.addOutput(attention_scores);
  fusion.addOutput(attention_probs);
  fusion.addOutput(mask);
  fusion.addOutput(output);

  std::vector<TensorView*> reduction_tensors;
  std::vector<TensorView*> other_tensors;
  analyzeFusion(&fusion, reduction_tensors, other_tensors);

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor at_scores = at::randn(input_shape, options);
  at::Tensor at_mask = at::randn(input_shape, options);
  std::vector<c10::IValue> inputs(
      {at_scores, at_mask, sqrt(kAttentionHeadSize)});

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

static void MagicScheduler_Softmax_Dropout_Baseline(
    benchmark::State& benchmark_state) {
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};
  const int kReductionAxis = 3;

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr float kDropoutProbability = 0.1;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;

  // inputs
  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor attention_scores = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    // Create
    CudaKernelTimer timer;

    // Run
    attention_scores = attention_scores / sqrt(kAttentionHeadSize);
    attention_scores = attention_scores + at_y;
    auto attention_probs =
        at::_softmax(attention_scores, kReductionAxis, false);
    attention_probs = at::dropout(attention_probs, kDropoutProbability, true);

    // Record
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
  }
}

BENCHMARK(MagicScheduler_Softmax_Dropout)
    ->Arg(8)
    ->Arg(16)
    ->Arg(24)
    ->Arg(32)
    ->Arg(40)
    ->Arg(48)
    ->Arg(56)
    ->Arg(64)
    ->Arg(72)
    ->Arg(80)
    ->Arg(88)
    ->Arg(96)
    ->Arg(104)
    ->Arg(112)
    ->Arg(120)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_Softmax_Dropout_Baseline)
    ->Arg(8)
    ->Arg(16)
    ->Arg(24)
    ->Arg(32)
    ->Arg(40)
    ->Arg(48)
    ->Arg(56)
    ->Arg(64)
    ->Arg(72)
    ->Arg(80)
    ->Arg(88)
    ->Arg(96)
    ->Arg(104)
    ->Arg(112)
    ->Arg(120)
    ->Arg(128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
