#include <torch/csrc/jit/codegen/cuda/arith.h>
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

static void setupSoftmaxDropout(
    Fusion* fusion,
    DataType dtype,
    const int kReductionAxis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  // setup fusion
  auto attention_scores = makeContigTensor(4, dtype);
  auto attention_mask = makeContigTensor(4, dtype);

  Double* divisor = IrBuilder::create<Double>();

  fusion->addInput(attention_scores);
  fusion->addInput(attention_mask);
  fusion->addInput(divisor);

  if (dtype == DataType::Half) {
    attention_scores = castOp(DataType::Float, attention_scores);
    attention_mask = castOp(DataType::Float, attention_mask);
  }

  attention_scores = div(attention_scores, divisor);
  attention_scores = add(attention_scores, attention_mask);
  auto attention_probs = softmax(attention_scores, kReductionAxis);
  auto prob = IrBuilder::create<Double>(kDropoutProbability);
  auto scale = IrBuilder::create<Double>(kScale);
  auto dropout_results = dropout(attention_probs, prob, scale);
  auto output = dropout_results.output;

  if (dtype == DataType::Half) {
    attention_scores = castOp(DataType::Half, attention_scores);
    attention_probs = castOp(DataType::Half, attention_probs);
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(attention_scores);
  fusion->addOutput(attention_probs);
  fusion->addOutput(output);

  fusion->addOutput(dropout_results.mask);
}

static void NvFuserScheduler_SoftmaxDropout(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int kReductionAxis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  // reduce across 1, [256, 12, 100, 8]
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;
  constexpr float kDropoutProbability = 0.9;
  constexpr float kScale = 1.0f / kDropoutProbability;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_scores = at::randn(input_shape, options);
  at::Tensor at_mask = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs(
      {at_scores, at_mask, sqrt(kAttentionHeadSize)});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  // 5 dtype: attention_scores + attention_mask + attention_scores_out +
  // attention_probs_out + output
  // 1 bool: dropout_results.mask
  // All the same size
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * 5 * at_scores.numel() *
          int64_t(dataTypeSize(dtype)) +
      // bool mask
      int64_t(benchmark_state.iterations()) * at_scores.numel() *
          int64_t(dataTypeSize(DataType::Bool)));
}

//------------------------------------------------------------------------------

static void Baseline_Softmax_Dropout(
    benchmark::State& benchmark_state,
    const int kReductionAxis,
    DataType dtype) {
  std::vector<int64_t> input_shape{256, 12, 100, benchmark_state.range(0)};

  constexpr int kHiddenSize = 768;
  constexpr int kNumAttentionHeads = 12;
  constexpr float kDropoutProbability = 0.1;
  constexpr int kAttentionHeadSize = kHiddenSize / kNumAttentionHeads;

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor attention_scores = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  for (auto _ : benchmark_state) {
    clearL2Cache();
    CudaKernelTimer timer;

    attention_scores = attention_scores / sqrt(kAttentionHeadSize);
    attention_scores = attention_scores + at_y;
    auto attention_probs =
        at::_softmax(attention_scores, kReductionAxis, false);
    attention_probs = at::dropout(attention_probs, kDropoutProbability, true);

    // Record
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // 5 dtype: attention_scores + attention_mask + attention_scores_out +
  // attention_probs_out + output
  // 1 bool: dropout_results.mask
  // All the same size
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * 5 * attention_scores.numel() *
          int64_t(dataTypeSize(dtype)) +
      // bool mask
      int64_t(benchmark_state.iterations()) * attention_scores.numel() *
          int64_t(dataTypeSize(DataType::Bool)));
}

//------------------------------------------------------------------------------

static void Baseline_Softmax_Dropout_Inner_fp32(
    benchmark::State& benchmark_state) {
  Baseline_Softmax_Dropout(benchmark_state, 3, DataType::Float);
}

static void Baseline_Softmax_Dropout_Outer_fp32(
    benchmark::State& benchmark_state) {
  Baseline_Softmax_Dropout(benchmark_state, 1, DataType::Float);
}

static void Baseline_Softmax_Dropout_Inner_fp16(
    benchmark::State& benchmark_state) {
  Baseline_Softmax_Dropout(benchmark_state, 3, DataType::Half);
}

static void Baseline_Softmax_Dropout_Outer_fp16(
    benchmark::State& benchmark_state) {
  Baseline_Softmax_Dropout(benchmark_state, 1, DataType::Half);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Dropout_Inner_fp32,
    setupSoftmaxDropout,
    NvFuserScheduler_SoftmaxDropout,
    DataType::Float,
    3);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Dropout_Inner_fp32)
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

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Dropout_Outer_fp32,
    setupSoftmaxDropout,
    NvFuserScheduler_SoftmaxDropout,
    DataType::Float,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Dropout_Outer_fp32)
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

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Dropout_Inner_fp16,
    setupSoftmaxDropout,
    NvFuserScheduler_SoftmaxDropout,
    DataType::Half,
    3);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Dropout_Inner_fp16)
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

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Dropout_Outer_fp16,
    setupSoftmaxDropout,
    NvFuserScheduler_SoftmaxDropout,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Dropout_Outer_fp16)
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

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Softmax_Dropout_Inner_fp32)
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

BENCHMARK(Baseline_Softmax_Dropout_Outer_fp32)
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

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Softmax_Dropout_Inner_fp16)
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

BENCHMARK(Baseline_Softmax_Dropout_Outer_fp16)
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
