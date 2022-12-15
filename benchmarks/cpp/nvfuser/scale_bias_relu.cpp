#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

static void setupSBR(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const size_t kNumberOfDims = 4;

  std::vector<int64_t> bcast_shape(kNumberOfDims, 1);
  bcast_shape[bcast_shape.size() - 1] = -1;

  std::vector<bool> bcast_contig(kNumberOfDims, false);
  bcast_contig[bcast_contig.size() - 1] = true;

  auto x = makeContigTensor(kNumberOfDims, dtype);

  auto scale = TensorViewBuilder()
                   .contiguity(bcast_contig)
                   .shape(bcast_shape)
                   .dtype(dtype)
                   .build();

  auto bias = TensorViewBuilder()
                  .contiguity(bcast_contig)
                  .shape(bcast_shape)
                  .dtype(dtype)
                  .build();

  fusion->addInput(x);
  fusion->addInput(scale);
  fusion->addInput(bias);

  if (dtype == DataType::Half) {
    x = castOp(DataType::Float, x);
    scale = castOp(DataType::Float, scale);
    bias = castOp(DataType::Float, bias);
  }

  auto scale_bias = add(mul(x, scale), bias);
  auto scale_bias_relu = unaryOp(UnaryOpType::Relu, scale_bias);

  if (dtype == DataType::Half) {
    scale_bias_relu = castOp(DataType::Half, scale_bias_relu);
  }
  fusion->addOutput(scale_bias_relu);
}

static void setupSBRNorm(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);
  FusionGuard fg(fusion);

  const size_t kNumberOfDims = 4;

  auto x = makeContigTensor(kNumberOfDims, dtype);
  auto weight = makeContigTensor(1, dtype);
  auto bias = makeContigTensor(1, dtype);
  auto mean = makeContigTensor(1, dtype);
  auto var = makeContigTensor(1, dtype);

  fusion->addInput(x);
  fusion->addInput(weight);
  fusion->addInput(bias);
  fusion->addInput(mean);
  fusion->addInput(var);

  std::vector<bool> broadcast_mask(kNumberOfDims, true);
  broadcast_mask[broadcast_mask.size() - 1] = false;

  if (dtype == DataType::Half) {
    x = castOp(DataType::Float, x);
    weight = castOp(DataType::Float, weight);
    bias = castOp(DataType::Float, bias);
    mean = castOp(DataType::Float, mean);
    var = castOp(DataType::Float, var);
  }

  auto rsqrt = unaryOp(UnaryOpType::Rsqrt, var);
  auto this_scale = mul(weight, rsqrt);
  auto this_bias = mul(sub(bias, mean), this_scale);

  auto bcast_scale = broadcast(this_scale, broadcast_mask);
  auto bcast_bias = broadcast(this_bias, broadcast_mask);

  auto scale_bias = add(mul(x, bcast_scale), bcast_bias);
  auto scale_bias_relu = unaryOp(UnaryOpType::Relu, scale_bias);

  if (dtype == DataType::Half) {
    scale_bias_relu = castOp(DataType::Half, scale_bias_relu);
  }

  fusion->addOutput(scale_bias_relu);
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_SBR(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{1, 1, 1, -1};

  // inputs
  at::manual_seed(0);
  std::vector<int64_t> static_bcast_shape{1, 1, 1, benchmark_state.range(2)};
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_scale = at::ones(static_bcast_shape, options);
  at::Tensor at_bias = at::zeros(static_bcast_shape, options);

  // inputs
  std::vector<c10::IValue> aten_inputs = {at_x, at_scale, at_bias};

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs(aten_inputs);

  auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
  auto executor_instance = compile_log.fusion_executor;
  auto params = toString(compile_log.params);
  auto lparams = toString(compile_log.fusion_executor->lastLaunchParams());

  benchmark_state.SetLabel(params + lparams);
  benchmark_state.SetLabel(lparams);

  fusion_executor_cache->profile(false);
  executor_instance->setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_SBR(benchmark::State& benchmark_state, DataType dtype) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_y = at::randn(input_shape, options);
  at::Tensor at_scale = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);

  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto scale = at::mul(at_x, at_scale);
    auto bias = at::add(scale, at_bias);
    auto output = at::relu(bias);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 2 + size * 2) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void NvFuserScheduler_SBR_Norm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);
  at::Tensor at_mean = at::zeros(bcast_shape, options);
  at::Tensor at_var = at::ones(bcast_shape, options);

  // inputs
  std::vector<c10::IValue> aten_inputs = {
      at_x, at_weight, at_bias, at_mean, at_var};

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs(aten_inputs);

  auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
  auto executor_instance = compile_log.fusion_executor;
  auto params = toString(compile_log.params);
  auto lparams = toString(compile_log.fusion_executor->lastLaunchParams());

  benchmark_state.SetLabel(params + lparams);

  fusion_executor_cache->profile(false);
  executor_instance->setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }

  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 4 + size * 2) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_SBR_Norm(
    benchmark::State& benchmark_state,
    DataType dtype) {
  // N, H, W, C format
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(1),
      benchmark_state.range(2)};
  std::vector<int64_t> bcast_shape{1, 1, 1, benchmark_state.range(2)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor at_x = at::randn(input_shape, options);
  at::Tensor at_weight = at::ones(bcast_shape, options);
  at::Tensor at_bias = at::zeros(bcast_shape, options);
  at::Tensor at_mean = at::zeros(bcast_shape, options);
  at::Tensor at_var = at::ones(bcast_shape, options);

  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    auto this_scale = at::mul(at_weight, at::rsqrt(at_var));
    auto this_bias = at::mul(at::sub(at_bias, at_mean), this_scale);

    auto scale = at::mul(at_x, this_scale);
    auto bias = at::add(scale, this_bias);
    auto output = at::relu(bias);

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  const size_t size =
      input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3];
  const size_t channels = input_shape[3];
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * (channels * 4 + size * 2) *
      int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SBR_fp32,
    setupSBR,
    NvFuserScheduler_SBR,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SBR_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SBR_fp16,
    setupSBR,
    NvFuserScheduler_SBR,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SBR_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SBR_Norm_fp32,
    setupSBRNorm,
    NvFuserScheduler_SBR_Norm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SBR_Norm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_SBR_Norm_fp16,
    setupSBRNorm,
    NvFuserScheduler_SBR_Norm,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_SBR_Norm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void Baseline_SBR_fp32(benchmark::State& benchmark_state) {
  Baseline_SBR(benchmark_state, DataType::Float);
}

BENCHMARK(Baseline_SBR_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void Baseline_SBR_fp16(benchmark::State& benchmark_state) {
  Baseline_SBR(benchmark_state, DataType::Half);
}

BENCHMARK(Baseline_SBR_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void Baseline_SBR_Norm_fp32(benchmark::State& benchmark_state) {
  Baseline_SBR_Norm(benchmark_state, DataType::Float);
}

BENCHMARK(Baseline_SBR_Norm_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

static void Baseline_SBR_Norm_fp16(benchmark::State& benchmark_state) {
  Baseline_SBR_Norm(benchmark_state, DataType::Half);
}

BENCHMARK(Baseline_SBR_Norm_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {640, 640}, {64, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
