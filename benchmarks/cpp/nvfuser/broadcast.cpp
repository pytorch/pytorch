#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <sstream>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

// Return broadcast tensor view and output of broadcast
static void setupBroadcast(Fusion* fusion, DataType dtype, int bcast_axis) {
  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = makeContigTensor(2, dtype);
  TensorView* tv1 = makeContigTensor(1, dtype);

  fusion->addInput(tv0);
  fusion->addInput(tv1);

  std::vector<bool> bcast_pattern(2, false);
  bcast_pattern[bcast_axis] = true;

  if (is_fp16) {
    tv0 = castOp(DataType::Float, tv0);
    tv1 = castOp(DataType::Float, tv1);
  }

  TensorView* tv2 = broadcast(tv1, bcast_pattern);
  TensorView* tv3 = add(tv0, tv2);

  if (is_fp16) {
    tv3 = castOp(DataType::Half, tv3);
  }

  fusion->addOutput(tv3);
}

static void NvFuserScheduler_Broadcast(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int bcast_dim) {
  auto bcast_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (bcast_dim ? at::randn({iter_size, bcast_size}, options)
                 : at::randn({bcast_size, iter_size}, options));

  at::Tensor t1 = at::randn({iter_size}, options);

  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs({t0, t1});

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
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs({t0, t1});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * bcast_size * 2 + iter_size) * int64_t(dataTypeSize(dtype)));
}

static void Baseline_Broadcast(
    benchmark::State& benchmark_state,
    DataType dtype,
    int bcast_dim) {
  auto bcast_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (bcast_dim ? at::randn({iter_size, bcast_size}, options)
                 : at::randn({bcast_size, iter_size}, options));

  at::Tensor t1 = at::randn({iter_size}, options);

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = t0.add(t1.unsqueeze(bcast_dim));
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * bcast_size * 2 + iter_size) * int64_t(dataTypeSize(dtype)));
}

//------------------------------------------------------------------------------

static void Baseline_Broadcast_Outer_fp32(benchmark::State& benchmark_state) {
  Baseline_Broadcast(benchmark_state, DataType::Float, 0);
}

static void Baseline_Broadcast_Outer_fp16(benchmark::State& benchmark_state) {
  Baseline_Broadcast(benchmark_state, DataType::Half, 0);
}

static void Baseline_Broadcast_Inner_fp32(benchmark::State& benchmark_state) {
  Baseline_Broadcast(benchmark_state, DataType::Float, 1);
}

static void Baseline_Broadcast_Inner_fp16(benchmark::State& benchmark_state) {
  Baseline_Broadcast(benchmark_state, DataType::Half, 1);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Broadcast_Outer_fp32,
    setupBroadcast,
    NvFuserScheduler_Broadcast,
    DataType::Float,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Broadcast_Outer_fp16,
    setupBroadcast,
    NvFuserScheduler_Broadcast,
    DataType::Half,
    0);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Broadcast_Inner_fp32,
    setupBroadcast,
    NvFuserScheduler_Broadcast,
    DataType::Float,
    1);
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Broadcast_Inner_fp16,
    setupBroadcast,
    NvFuserScheduler_Broadcast,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Broadcast_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
