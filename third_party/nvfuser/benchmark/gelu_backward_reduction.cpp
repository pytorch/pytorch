#include <executor.h>
#include <fusion.h>
#include <ir_all_nodes.h>
#include <ir_utils.h>
#include <lower2device.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include <benchmark/utils.h>

// Return reduction tensor view and output of reduction
static void setupGeluBackwardReduction(
    Fusion* fusion,
    DataType dtype,
    int red_axis) {
  FusionGuard fg(fusion);

  constexpr float k_079 = 0.79788456;
  constexpr float k_004 = 0.044715;
  constexpr float k_010 = 0.1070322243;

  // input fp16 is converted to fp32 before caluclation and converted back to
  // fp16 after calculation
  bool is_fp16 = dtype == DataType::Half;

  // gradient tensor
  auto t0 = makeContigTensor(2, dtype);
  fusion->addInput(t0);
  auto t1 = t0;
  if (is_fp16)
    t1 = castOp(DataType::Float, t0);

  // input tensor
  auto t4 = makeContigTensor(2, dtype);
  fusion->addInput(t4);
  auto t5 = t4;
  if (is_fp16)
    t5 = castOp(DataType::Float, t4);

  // calc-1, gelu backward
  auto t7 = castOp(DataType::Float, t5);
  auto t8 = mul(t7, IrBuilder::create<Double>(k_079));
  auto t9 = mul(t7, IrBuilder::create<Double>(k_004));
  auto t10 = mul(t9, t7);
  auto t11 = add(t10, IrBuilder::create<Int>(1));
  auto t12 = mul(t8, t11);
  auto t13 = unaryOp(UnaryOpType::Tanh, t12);
  auto t14 = mul(t7, IrBuilder::create<Double>(0.5));
  auto t15 = mul(t13, t13);
  auto t16 = unaryOp(UnaryOpType::Neg, t15);
  auto t17 = add(t16, IrBuilder::create<Int>(1));
  auto t18 = mul(t7, IrBuilder::create<Double>(k_010));
  auto t19 = mul(t18, t7);
  auto t20 = add(t19, IrBuilder::create<Double>(k_079));
  auto t21 = mul(t17, t20);
  auto t22 = mul(t14, t21);
  auto t23 = add(t13, IrBuilder::create<Int>(1));
  auto t24 = mul(t23, IrBuilder::create<Double>(0.5));
  auto t25 = add(t22, t24);
  auto t26 = mul(t25, t1);

  // output of gelu backward
  auto t27 = t26;
  if (is_fp16)
    t27 = castOp(dtype, t26);
  fusion->addOutput(t27);

  // calc-2, reduction
  auto t26r = sum(t26, {red_axis});

  // output of reduction
  auto t28 = t26r;
  if (is_fp16)
    t28 = castOp(dtype, t26r);
  fusion->addOutput(t28);
}

static void NvFuserScheduler_GeluBackwardReduction(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int reduction_dim) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input_x =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
                     : at::randn({reduction_size, iter_size}, options));

  at::Tensor aten_input_grad =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
                     : at::randn({reduction_size, iter_size}, options));

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs({aten_input_grad, aten_input_x});

  auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
  auto executor_instance = compile_log.fusion_executor;
  auto rparams = toString(compile_log.params);
  auto lparams = toString(compile_log.fusion_executor->lastLaunchParams());

  benchmark_state.SetLabel(rparams + lparams);

  fusion_executor_cache->profile(false);
  executor_instance->setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs(
        {aten_input_grad, aten_input_x});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size) * int64_t(dataTypeSize(dtype)));
}

static void Baseline_GeluBackwardReduction(
    benchmark::State& benchmark_state,
    DataType dtype,
    int reduction_dim) {
  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  constexpr float k_079 = 0.79788456;
  constexpr float k_004 = 0.044715;
  constexpr float k_010 = 0.1070322243;
  constexpr bool use_fused = true;
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto tensor_shape = reduction_dim
      ? std::vector<int64_t>{iter_size, reduction_size}
      : std::vector<int64_t>{reduction_size, iter_size};

  at::Tensor at_grad = at::randn(tensor_shape, options);
  at::Tensor at_xvar = at::randn(tensor_shape, options);

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto at_x = at_xvar;
    if (dtype == DataType::Half)
      at_x = at_xvar.to(c10::ScalarType::Float);

    if (!use_fused) {
      auto at_tanh_out = (k_079 * at_x * (1 + k_004 * at_x * at_x)).tanh();
      auto at_ff = 0.5 * at_x *
              ((1 - at_tanh_out * at_tanh_out) *
               (k_079 + k_010 * at_x * at_x)) +
          0.5 * (1 + at_tanh_out);
      auto at_output_pointwise = at_ff * at_grad;
      auto at_output_reduction = at_output_pointwise.sum({0});
    } else {
      auto at_output_pointwise = at::gelu_backward(at_grad, at_x, "tanh");
      auto at_output_reduction = at_output_pointwise.sum({0});
    }

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size) * int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void Baseline_GeluBackwardReduction_Outer_fp32(
    benchmark::State& benchmark_state) {
  Baseline_GeluBackwardReduction(benchmark_state, DataType::Float, 0);
}

static void Baseline_GeluBackwardReduction_Outer_fp16(
    benchmark::State& benchmark_state) {
  Baseline_GeluBackwardReduction(benchmark_state, DataType::Half, 0);
}

static void Baseline_GeluBackwardReduction_Inner_fp32(
    benchmark::State& benchmark_state) {
  Baseline_GeluBackwardReduction(benchmark_state, DataType::Float, 1);
}

static void Baseline_GeluBackwardReduction_Inner_fp16(
    benchmark::State& benchmark_state) {
  Baseline_GeluBackwardReduction(benchmark_state, DataType::Half, 1);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_GeluBackwardReduction_Outer_fp32,
    setupGeluBackwardReduction,
    NvFuserScheduler_GeluBackwardReduction,
    DataType::Float,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_GeluBackwardReduction_Outer_fp16,
    setupGeluBackwardReduction,
    NvFuserScheduler_GeluBackwardReduction,
    DataType::Half,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_GeluBackwardReduction_Inner_fp32,
    setupGeluBackwardReduction,
    NvFuserScheduler_GeluBackwardReduction,
    DataType::Float,
    1);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_GeluBackwardReduction_Inner_fp16,
    setupGeluBackwardReduction,
    NvFuserScheduler_GeluBackwardReduction,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_GeluBackwardReduction_Outer_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_GeluBackwardReduction_Outer_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_GeluBackwardReduction_Outer_fp32)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_GeluBackwardReduction_Outer_fp16)
    ->RangeMultiplier(2)
    ->Ranges({{512, 512 * 64}, {512, 512 * 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
