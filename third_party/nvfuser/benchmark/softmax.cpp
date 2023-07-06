#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/ops/all_ops.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/nvfuser/utils.h>

using namespace torch::jit::fuser::cuda;

//------------------------------------------------------------------------------

static void setupSoftmax(
    Fusion* fusion,
    DataType dtype,
    const int reduction_axis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  FusionGuard fg(fusion);
  // setup fusion
  auto input = makeContigTensor(2, dtype);
  fusion->addInput(input);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
  }

  auto output = softmax(input, reduction_axis);

  if (dtype == DataType::Half) {
    output = castOp(DataType::Half, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_Softmax(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    const int reduction_axis) {
  TORCH_INTERNAL_ASSERT(dtype == DataType::Float || dtype == DataType::Half);

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::Tensor aten_input =
      (reduction_axis ? at::randn({iter_size, reduction_size}, options)
                      : at::randn({reduction_size, iter_size}, options));

  std::vector<c10::IValue> aten_inputs({aten_input});

  runBenchmarkIterations(benchmark_state, fusion_executor_cache, aten_inputs);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

// Warp softmax comparison
static void Softmax_WarpReduceReference(benchmark::State& benchmark_state) {
  auto dtype = DataType::Float;
  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  setupSoftmax(fusion, dtype, 1);

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({aten_input});

  // Schedule through magic scheduler:
  SchedulerRuntimeInfo runtime_info(fusion, aten_inputs, true);
  TORCH_INTERNAL_ASSERT(SchedulerEntry::canSchedule(
      ScheduleHeuristic::Persistent, fusion, runtime_info));
  auto scheduler = SchedulerEntry::makeEntry(
      ScheduleHeuristic::Persistent, fusion, runtime_info);
  scheduler->schedule(fusion);

  FusionExecutor fe;
  fe.compileFusion(fusion);
  auto outputs = fe.runFusion(aten_inputs);
  fe.setMeasureKernelTimeFlag(true);

  // Sync everything up before we start
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto outputs = fe.runFusion(aten_inputs);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

static void Softmax_WarpReduce(benchmark::State& benchmark_state) {
  auto dtype = DataType::Float;
  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  setupSoftmax(fusion, dtype, 1);

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input = at::randn(input_shape, options);
  std::vector<c10::IValue> aten_inputs({aten_input});

  // Schedule through magic scheduler:
  SchedulerRuntimeInfo runtime_info(fusion, aten_inputs, true);
  TORCH_INTERNAL_ASSERT(SchedulerEntry::canSchedule(
      ScheduleHeuristic::Persistent, fusion, runtime_info));
  auto scheduler = SchedulerEntry::makeEntry(
      ScheduleHeuristic::Persistent, fusion, runtime_info);
  scheduler->schedule(fusion);

  // Modify the schedule to use warp reduction
  auto used_vals = fusion->usedMathVals();
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    for (IterDomain* id : tv->domain()->domain()) {
      if (id->getParallelType() == ParallelType::TIDx) {
        id->padToMultipleOfWarp();
      }
    }
  }

  FusionExecutor fe;
  fe.compileFusion(fusion);
  auto outputs = fe.runFusion(aten_inputs);
  fe.setMeasureKernelTimeFlag(true);

  // Sync everything up before we start
  for (auto _ : benchmark_state) {
    clearL2Cache();
    auto outputs = fe.runFusion(aten_inputs);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

BENCHMARK(Softmax_WarpReduce)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16 * 197, 16 * 197}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Softmax_WarpReduceReference)
    ->RangeMultiplier(2)
    ->Ranges({{8, 8}, {16 * 197, 16 * 197}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

static void Baseline_Softmax(
    benchmark::State& benchmark_state,
    DataType dtype,
    const int reduction_axis) {
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  at::Tensor aten_input =
      (reduction_axis ? at::randn({iter_size, reduction_size}, options)
                      : at::randn({reduction_size, iter_size}, options));

  for (auto _ : benchmark_state) {
    clearL2Cache();
    CudaKernelTimer timer;
    auto output = at::_softmax(aten_input, reduction_axis, false);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * aten_input.numel() * int64_t(dataTypeSize(dtype))));
}

static void Baseline_Softmax_Outer_fp32(benchmark::State& benchmark_state) {
  Baseline_Softmax(benchmark_state, DataType::Float, 0);
}

static void Baseline_Softmax_Inner_fp32(benchmark::State& benchmark_state) {
  Baseline_Softmax(benchmark_state, DataType::Float, 1);
}

static void Baseline_Softmax_Outer_fp16(benchmark::State& benchmark_state) {
  Baseline_Softmax(benchmark_state, DataType::Half, 0);
}

static void Baseline_Softmax_Inner_fp16(benchmark::State& benchmark_state) {
  Baseline_Softmax(benchmark_state, DataType::Half, 1);
}

//------------------------------------------------------------------------------

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Outer_fp32,
    setupSoftmax,
    NvFuserScheduler_Softmax,
    DataType::Float,
    0);

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Inner_fp32,
    setupSoftmax,
    NvFuserScheduler_Softmax,
    DataType::Float,
    1);

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Outer_fp16,
    setupSoftmax,
    NvFuserScheduler_Softmax,
    DataType::Half,
    0);

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Softmax_Inner_fp16,
    setupSoftmax,
    NvFuserScheduler_Softmax,
    DataType::Half,
    1);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

//------------------------------------------------------------------------------

BENCHMARK(Baseline_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Outer_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {32768, 32 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_Softmax_Inner_fp16)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 1024 * 16}, {128, 1024 * 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
