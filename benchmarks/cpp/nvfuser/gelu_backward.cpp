
// Based on NVFuserTest.FusionBiasGeluBwd_CUDA

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  const float k_079 = 0.79788456;
  const float k_004 = 0.044715;
  const float k_010 = 0.1070322243;

  // gradient tensor
  auto t0 = TensorViewBuilder().ndims(3).dtype(DataType::Half).build();
  fusion->addInput(t0);

  auto t1 = castOp(DataType::Float, t0);

  // bias tensor
  auto t2 = TensorViewBuilder().ndims(1).dtype(DataType::Half).build();
  fusion->addInput(t2);

  auto t3 = castOp(DataType::Float, t2);
  
  // input tensor
  auto t4 = TensorViewBuilder().ndims(3).dtype(DataType::Half).build();
  fusion->addInput(t4);

  auto t5 = castOp(DataType::Float, t4);
  auto t6 = broadcast(t3, {true, true, false});
  auto t7 = add(t6, t5);
  auto t8 = mul(t7, new Float(k_079));
  auto t9 = mul(t7, new Float(k_004));
  auto t10 = mul(t9, t7);
  auto t11 = add(t10, new Int(1));
  auto t12 = mul(t8, t11);
  auto t13 = unaryOp(UnaryOpType::Tanh, t12);
  auto t14 = mul(t7, new Float(0.5));
  auto t15 = mul(t13, t13);
  auto t16 = unaryOp(UnaryOpType::Neg, t15);
  auto t17 = add(t16, new Int(1));
  auto t18 = mul(t7, new Float(k_010));
  auto t19 = mul(t18, t7);
  auto t20 = add(t19, new Float(k_079));
  auto t21 = mul(t17, t20);
  auto t22 = mul(t14, t21);
  auto t23 = add(t13, new Int(1));
  auto t24 = mul(t23, new Float(0.5));
  auto t25 = add(t22, t24);
  auto t26 = mul(t25, t1);
  
  // Save float output for validation
  fusion->addOutput(t26);
  auto t27 = castOp(DataType::Half, t26);
  fusion->addOutput(t27);
}

static std::vector<c10::IValue> setupInputs() {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  c10::IntArrayRef input_shape{6, 512, 4096};
  c10::IntArrayRef bias_shape{4096};
  auto at_input = at::randn(input_shape, options);
  auto at_bias = at::randn(bias_shape, options);
  auto at_grad = at::randn(input_shape, options);

  return {at_grad, at_bias, at_input};
}

//------------------------------------------------------------------------------

static void GeluBackward_SetupFusion(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(GeluBackward_SetupFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void GeluBackward_AutoSchedule(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    std::vector<c10::IValue> inputs = setupInputs();
    benchmark_state.ResumeTiming();

    // Auto-schedule
    scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK(GeluBackward_AutoSchedule)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void GeluBackward_Lower(benchmark::State& benchmark_state) {
  constexpr int kHiddenFeatures = 512;
  constexpr int kBatchSize = 64;

  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    GpuLower gpu_lower(&fusion);
  }
}

BENCHMARK(GeluBackward_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void GeluBackward_Compile(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    FusionExecutor executor;
    executor.compileFusion(&fusion);
  }
}

BENCHMARK(GeluBackward_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void GeluBackward_RunFusion(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  // outputs
  std::vector<at::Tensor> outputs;

  scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.compileFusion(&fusion);

  cudaDeviceSynchronize();
  
  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
    cudaDeviceSynchronize();
  }
}

BENCHMARK(GeluBackward_RunFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void GeluBackward_RunFusion_CpuOnly(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  // outputs
  std::vector<at::Tensor> outputs;

  scheduleFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.setExecuteKernelFlag(false);
  executor.compileFusion(&fusion);

  for (auto _ : benchmark_state) {
    outputs = executor.runFusion(c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK(GeluBackward_RunFusion_CpuOnly)->Unit(benchmark::kMicrosecond);
