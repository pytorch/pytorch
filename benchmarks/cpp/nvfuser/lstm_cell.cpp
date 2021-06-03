#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

// TODO: add LSTM function to composite operations
// Function Signature: cy, hy = lstm(x, cx)
static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  TensorView* tvs[16];
  for (size_t i = 0; i < 16; i++) {
    tvs[i] = TensorViewBuilder().ndims(2).dtype(DataType::Float).build();
    fusion->addInput(tvs[i]);
  }

  const auto ingate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[0], tvs[1]), tvs[2]), tvs[3]));

  const auto forgetgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[4], tvs[5]), tvs[6]), tvs[7]));

  const auto cellgate = unaryOp(
      UnaryOpType::Tanh, add(add(add(tvs[8], tvs[9]), tvs[10]), tvs[11]));

  const auto outgate = unaryOp(
      UnaryOpType::Sigmoid, add(add(add(tvs[12], tvs[13]), tvs[14]), tvs[15]));

  const auto cx = TensorViewBuilder()
                      .ndims(2)
                      .dtype(DataType::Float)
                      .contiguity(std::vector<bool>(2, true))
                      .build();

  const auto cy = add(mul(forgetgate, cx), mul(ingate, cellgate));

  const auto hy = mul(outgate, unaryOp(UnaryOpType::Tanh, cy));

  fusion->addInput(cx);
  fusion->addOutput(cy);
  fusion->addOutput(hy);
}

static std::vector<c10::IValue> setupInputs(
    int hidden_features,
    int batch_size) {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  const at::Tensor large_tensor0 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor1 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor2 =
      at::randn({batch_size, hidden_features * 4}, options);
  const at::Tensor large_tensor3 =
      at::randn({batch_size, hidden_features * 4}, options);

  const auto chunked0 = large_tensor0.chunk(4, 1);
  const auto chunked1 = large_tensor1.chunk(4, 1);
  const auto chunked2 = large_tensor2.chunk(4, 1);
  const auto chunked3 = large_tensor3.chunk(4, 1);

  std::vector<c10::IValue> inputs;
  inputs.insert(inputs.end(), chunked0.begin(), chunked0.end());
  inputs.insert(inputs.end(), chunked1.begin(), chunked1.end());
  inputs.insert(inputs.end(), chunked2.begin(), chunked2.end());
  inputs.insert(inputs.end(), chunked3.begin(), chunked3.end());

  const auto at_cx = at::randn({batch_size, hidden_features}, options);
  inputs.push_back(at_cx);

  return inputs;
}

//------------------------------------------------------------------------------

static void LstmCell_SetupFusion(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(LstmCell_SetupFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void LstmCell_AutoSchedule(benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    std::vector<c10::IValue> inputs = setupInputs(kHiddenFeatures, kBatchSize);
    benchmark_state.ResumeTiming();

    // Auto-schedule
    schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK(LstmCell_AutoSchedule)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void LstmCell_Lower(benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(kHiddenFeatures, kBatchSize);

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    GpuLower gpu_lower(&fusion);
  }
}

BENCHMARK(LstmCell_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void LstmCell_Compile(benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(kHiddenFeatures, kBatchSize);

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    FusionExecutor executor;
    executor.compileFusion(&fusion);
  }
}

BENCHMARK(LstmCell_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void LstmCell_RunFusion(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(hidden_features, batch_size);

  // outputs
  std::vector<at::Tensor> outputs;

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
    cudaDeviceSynchronize();
  }
}

BENCHMARK_CAPTURE(LstmCell_RunFusion, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(LstmCell_RunFusion, Medium, 1024, 128)
    ->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void LstmCell_RunFusion_GpuOnly(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(hidden_features, batch_size);

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
}

BENCHMARK_CAPTURE(LstmCell_RunFusion_GpuOnly, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK_CAPTURE(LstmCell_RunFusion_GpuOnly, Medium, 1024, 128)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void LstmCell_RunFusion_CpuOnly(
    benchmark::State& benchmark_state,
    int hidden_features,
    int batch_size) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs(hidden_features, batch_size);

  // outputs
  std::vector<at::Tensor> outputs;

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.setExecuteKernelFlag(false);
  executor.compileFusion(&fusion);

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK_CAPTURE(LstmCell_RunFusion_CpuOnly, Small, 512, 64)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_CAPTURE(LstmCell_RunFusion_CpuOnly, Medium, 1024, 128)
    ->Unit(benchmark::kMicrosecond);
