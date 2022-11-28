#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <ATen/Operators.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupBatchNorm_BWD(Fusion* fusion, DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);

  const bool kTraining = true;
  const float kEps = 1e-5;

  // setup fusion
  auto input = makeContigTensor(4, dtype);
  auto grad_output = makeContigTensor(4, dtype);
  auto weight = makeContigTensor(1, DataType::Float);
  auto running_mean = makeContigTensor(1, DataType::Float);
  auto running_var = makeContigTensor(1, DataType::Float);
  auto save_mean = makeContigTensor(1, DataType::Float);
  auto save_var = makeContigTensor(1, DataType::Float);

  fusion->addInput(input);
  fusion->addInput(grad_output);
  fusion->addInput(weight);
  fusion->addInput(running_mean);
  fusion->addInput(running_var);
  fusion->addInput(save_mean);
  fusion->addInput(save_var);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    grad_output = castOp(DataType::Float, grad_output);
  }

  auto eps_ptr = IrBuilder::create<Double>(kEps);

  auto result = batch_norm_backward(
      input,
      grad_output,
      weight,
      running_mean,
      running_var,
      save_mean,
      save_var,
      kTraining,
      eps_ptr,
      std::vector<bool>(3, true));

  auto grad_input = result.grad_input;
  auto grad_weight = result.grad_weight;
  auto grad_bias = result.grad_bias;

  if (dtype == DataType::Half) {
    grad_input = castOp(DataType::Half, grad_input);
    grad_weight = castOp(DataType::Half, grad_weight);
    grad_bias = castOp(DataType::Half, grad_bias);
  }

  fusion->addOutput(grad_input);
  fusion->addOutput(grad_weight);
  fusion->addOutput(grad_bias);
}

static void NvFuserScheduler_BatchNorm_BWD(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor grad_out = at::randn(input_shape, options);
  at::Tensor weight = at::ones({input_shape[1]}, fp32_options);
  at::Tensor run_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor run_var = at::ones({input_shape[1]}, fp32_options);
  at::Tensor save_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor save_var = at::ones({input_shape[1]}, fp32_options);

  std::vector<c10::IValue> aten_inputs(
      {input, grad_out, weight, run_mean, run_var, save_mean, save_var});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (((3 * input.numel()) * int64_t(dataTypeSize(dtype))) +
       (run_mean.numel() + run_var.numel() + save_mean.numel() +
        save_var.numel() + weight.numel()) *
           int64_t(dataTypeSize(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_BWD(
    benchmark::State& benchmark_state,
    DataType dtype) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  const float kMomentum = 0.1;
  const float kEps = 1e-5;
  std::vector<int64_t> input_shape{
      benchmark_state.range(0),
      benchmark_state.range(1),
      benchmark_state.range(2),
      benchmark_state.range(2)};

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto fp32_options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor grad_out = at::randn(input_shape, options);
  at::Tensor weight = at::ones({input_shape[1]}, fp32_options);
  at::Tensor bias = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor run_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor run_var = at::ones({input_shape[1]}, fp32_options);
  at::Tensor save_mean = at::zeros({input_shape[1]}, fp32_options);
  at::Tensor save_var = at::ones({input_shape[1]}, fp32_options);

  auto ato_weight = c10::optional<at::Tensor>(weight);
  auto ato_bias = c10::optional<at::Tensor>(bias);
  auto ato_run_mean = c10::optional<at::Tensor>(run_mean);
  auto ato_run_var = c10::optional<at::Tensor>(run_var);
  auto ato_save_mean = c10::optional<at::Tensor>(save_mean);
  auto ato_save_var = c10::optional<at::Tensor>(save_var);

  auto fwd_result = at::_ops::_batch_norm_impl_index::call(
      input,
      ato_weight,
      ato_bias,
      ato_run_mean,
      ato_run_var,
      true,
      kMomentum,
      kEps,
      true);
  cudaDeviceSynchronize();

  // Sync everything up before we start
  clearL2Cache();
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;

    at::_ops::cudnn_batch_norm_backward::call(
        input,
        grad_out,
        weight,
        ato_run_mean,
        ato_run_var,
        save_mean,
        save_var,
        kEps,
        std::get<3>(fwd_result));

    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    cudaDeviceSynchronize();
    clearL2Cache();
    cudaDeviceSynchronize();
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (((3 * input.numel()) * int64_t(dataTypeSize(dtype))) +
       (run_mean.numel() + run_var.numel() + save_mean.numel() +
        save_var.numel() + weight.numel()) *
           int64_t(dataTypeSize(DataType::Float))));
}

//------------------------------------------------------------------------------

static void Baseline_BatchNorm_BWD_cuDNN_fp32(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_BWD(benchmark_state, DataType::Float);
}

static void Baseline_BatchNorm_BWD_cuDNN_fp16(
    benchmark::State& benchmark_state) {
  Baseline_BatchNorm_BWD(benchmark_state, DataType::Half);
}

// Simple aliases just for names in the printed output
static void Baseline_ResNet_BatchNorm_BWD_cuDNN_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm_BWD(benchmark_state, DataType::Half);
}

static void Baseline_ResNext_BatchNorm_BWD_cuDNN_fp16(benchmark::State& benchmark_state) {
  Baseline_BatchNorm_BWD(benchmark_state, DataType::Half);
}
//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_BWD_fp32,
    setupBatchNorm_BWD,
    NvFuserScheduler_BatchNorm_BWD,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_BWD_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_BatchNorm_BWD_fp16,
    setupBatchNorm_BWD,
    NvFuserScheduler_BatchNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_BatchNorm_BWD_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_BatchNorm_BWD_cuDNN_fp32)
    // ->RangeMultiplier(2)
    // cuDNN didn't make it to 1024
    ->Ranges({{64, 512}, {32, 128}, {2, 64}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_BWD_cuDNN_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_BWD_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{64, 512}, {32, 128}, {2, 128}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_BatchNorm_BWD_cuDNN_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 64}, {2, 32}, {2, 256}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------
// RESNET and REXNEXT benchmarks

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNet_BatchNorm_BWD_fp16,
    setupBatchNorm_BWD,
    NvFuserScheduler_BatchNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNet_BatchNorm_BWD_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_ResNext_BatchNorm_BWD_fp16,
    setupBatchNorm_BWD,
    NvFuserScheduler_BatchNorm_BWD,
    DataType::Half);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_ResNext_BatchNorm_BWD_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_ResNet_BatchNorm_BWD_cuDNN_fp16)
    ->Args({256, 64, 112})
    ->Args({256, 64, 56})
    ->Args({256, 256, 56})
    ->Args({256, 128, 56})
    ->Args({256, 128, 28})
    ->Args({256, 512, 28})
    ->Args({256, 256, 28})
    ->Args({256, 256, 14})
    ->Args({256, 1024, 14})
    ->Args({256, 512, 14})
    ->Args({256, 512, 7})
    ->Args({256, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_ResNext_BatchNorm_BWD_cuDNN_fp16)
    ->Args({128, 64, 112})
    ->Args({128, 128, 56})
    ->Args({128, 256, 56})
    ->Args({128, 128, 56})
    ->Args({128, 256, 28})
    ->Args({128, 512, 28})
    ->Args({128, 512, 14})
    ->Args({128, 1024, 14})
    ->Args({128, 1024, 7})
    ->Args({128, 2048, 7})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
