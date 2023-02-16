
// Based on NVFuserTest.FusionBiasGeluBwd_CUDA

#include <executor.h>
#include <fusion.h>
#include <ir_builder.h>
#include <lower2device.h>
#include <ops/arith.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmark/utils.h>

using namespace torch::jit::fuser::cuda;

static void setupFusion(Fusion* fusion) {
  FusionGuard fg(fusion);

  // set up input tensor views
  auto t0 = makeContigTensor(2); // nDim = 2
  // scaling tensor
  auto t1 = makeContigTensor(2);
  fusion->addInput(t1);
  fusion->addInput(t0);
  auto t_idx = makeContigTensor(1, DataType::Int);
  fusion->addInput(t_idx);

  auto t2 = index_select(t0, 0, t_idx); // select at dim=0
  auto t3 = mul(t1, t2);
  auto t4 = add(t3, IrBuilder::create<Double>(17.0));

  // Save float output for validation
  fusion->addOutput(t4);
}

static std::vector<c10::IValue> setupInputs() {
  at::manual_seed(0);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  int nElem = 1023;
  int nElem_select = nElem + 115;
  int nFeat = 128;
  std::vector<int64_t> input_shape{nElem, nFeat};
  std::vector<int64_t> select_shape{nElem_select, nFeat};
  auto at_input = at::randn(input_shape, options);
  auto at_select = at::randn(select_shape, options);
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  auto at_index = at::randint(nElem, {nElem_select}, indx_options);
  return {at_select, at_input, at_index};
}

//------------------------------------------------------------------------------

static void IndexSelect_SetupFusion(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    Fusion fusion;
    setupFusion(&fusion);
  }
}

BENCHMARK(IndexSelect_SetupFusion)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void IndexSelect_AutoSchedule(benchmark::State& benchmark_state) {
  for (auto _ : benchmark_state) {
    // Setup (not included in the measurement)
    benchmark_state.PauseTiming();
    Fusion fusion;
    setupFusion(&fusion);
    std::vector<c10::IValue> inputs = setupInputs();
    benchmark_state.ResumeTiming();

    // Auto-schedule
    schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));
  }
}

BENCHMARK(IndexSelect_AutoSchedule)->Unit(benchmark::kMicrosecond);

//------------------------------------------------------------------------------

static void IndexSelect_Lower(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    GpuLower gpu_lower(&fusion);
  }
}

BENCHMARK(IndexSelect_Lower)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void IndexSelect_Compile(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  for (auto _ : benchmark_state) {
    FusionExecutor executor;
    executor.compileFusion(
        &fusion, c10::ArrayRef<c10::IValue>(inputs), lparams);
  }
}

BENCHMARK(IndexSelect_Compile)->Unit(benchmark::kMillisecond);

//------------------------------------------------------------------------------

static void IndexSelect_RunFusion(benchmark::State& benchmark_state) {
  Fusion fusion;

  // setup fusion
  setupFusion(&fusion);

  // inputs
  std::vector<c10::IValue> inputs = setupInputs();

  auto lparams = schedulePointwise(&fusion, c10::ArrayRef<c10::IValue>(inputs));

  FusionExecutor executor;
  executor.compileFusion(&fusion, c10::ArrayRef<c10::IValue>(inputs), lparams);

  C10_CUDA_CHECK(cudaDeviceSynchronize());

  at::Tensor output = at::empty_like(inputs[0].toTensor());

  for (auto _ : benchmark_state) {
    executor.runFusion(c10::ArrayRef<c10::IValue>(inputs), {output}, lparams);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
  }
}

BENCHMARK(IndexSelect_RunFusion)->Unit(benchmark::kMicrosecond);

static void setupIndexSelectSimple(
    Fusion* fusion,
    DataType dtype,
    int select_dim) {
  FusionGuard fg(fusion);
  bool is_fp16 = dtype == DataType::Half;

  // set up input tensor views
  auto t0 = makeContigTensor(2, dtype); // nDim = 2
  // scaling tensor
  auto t1 = makeContigTensor(2, dtype);
  fusion->addInput(t0);
  if (is_fp16) {
    t0 = castOp(DataType::Float, t0);
    t1 = castOp(DataType::Float, t1);
  }

  auto t_idx = makeContigTensor(1, DataType::Int);
  fusion->addInput(t_idx);

  auto t2 = index_select(t0, select_dim, t_idx); // select at dim=0
  if (is_fp16) {
    t2 = castOp(DataType::Half, t2);
  }
  fusion->addOutput(t2);
}

static void setupIndexSelect(Fusion* fusion, DataType dtype, int select_dim) {
  FusionGuard fg(fusion);
  bool is_fp16 = dtype == DataType::Half;

  // set up input tensor views
  auto t0 = makeContigTensor(2, dtype); // nDim = 2
  // scaling tensor
  auto t1 = makeContigTensor(2, dtype);
  fusion->addInput(t1);
  fusion->addInput(t0);
  if (is_fp16) {
    t0 = castOp(DataType::Float, t0);
    t1 = castOp(DataType::Float, t1);
  }

  auto t_idx = makeContigTensor(1, DataType::Int);
  fusion->addInput(t_idx);

  auto t2 = index_select(t0, select_dim, t_idx); // select at dim=0
  auto t3 = mul(t1, t2);
  auto t4 = add(t3, IrBuilder::create<Double>(17.0));

  if (is_fp16) {
    t4 = castOp(DataType::Half, t4);
  }

  // Save float output for validation
  fusion->addOutput(t4);
}

static void NvFuserScheduler_IndexSelectSimple(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int select_dim) {
  auto elem_size = benchmark_state.range(0);
  auto select_size = benchmark_state.range(1);
  int nFeat = 128; // lets fix feat dim for now

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (select_dim ? at::randn({nFeat, elem_size}, options)
                  : at::randn({elem_size, nFeat}, options));
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randint(elem_size, {select_size}, indx_options);
  at::Tensor t2 =
      (select_dim ? at::randn({nFeat, select_size}, options)
                  : at::randn({select_size, nFeat}, options));
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
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs({t0, t1});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (select_size + nFeat * select_size /*index select op*/) *
      int64_t(dataTypeSize(dtype)));
}

static void NvFuserScheduler_IndexSelect(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    DataType dtype,
    int select_dim) {
  auto elem_size = benchmark_state.range(0);
  auto select_size = benchmark_state.range(1);
  int nFeat = 128; // lets fix feat dim for now

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (select_dim ? at::randn({nFeat, elem_size}, options)
                  : at::randn({elem_size, nFeat}, options));
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randint(elem_size, {select_size}, indx_options);
  at::Tensor t2 =
      (select_dim ? at::randn({nFeat, select_size}, options)
                  : at::randn({select_size, nFeat}, options));
  fusion_executor_cache->profile(true);
  fusion_executor_cache->runFusionWithInputs({t2, t0, t1});

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
    auto cg_outputs = fusion_executor_cache->runFusionWithInputs({t2, t0, t1});
    benchmark_state.SetIterationTime(
        executor_instance->kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (nFeat * select_size * 2 /*2 elemwise ops*/ + select_size +
       nFeat * select_size /*index select op*/) *
      int64_t(dataTypeSize(dtype)));
}

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_IndexSelectSimple_Outer_fp32,
    setupIndexSelectSimple,
    NvFuserScheduler_IndexSelectSimple,
    DataType::Float,
    0);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_IndexSelectSimple_Outer_fp32)
    ->Ranges({{32768, 32768}, {65536, 65536}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_IndexSelect_Outer_fp32,
    setupIndexSelect,
    NvFuserScheduler_IndexSelect,
    DataType::Float,
    0);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_IndexSelect_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 32768}, {16, 32768}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// -------------------------------------- Baseline model
// ------------------------------------------------------------
static void Baseline_IndexSelectSimple(
    benchmark::State& benchmark_state,
    DataType dtype,
    int select_dim) {
  auto elem_size = benchmark_state.range(0);
  auto select_size = benchmark_state.range(1);
  int nFeat = 128; // lets fix feat dim for now

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (select_dim ? at::randn({nFeat, elem_size}, options)
                  : at::randn({elem_size, nFeat}, options));
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randint(elem_size, {select_size}, indx_options);

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::index_select(t0, select_dim, t1);
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (select_size + nFeat * select_size /*index select op*/) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_IndexSelect(
    benchmark::State& benchmark_state,
    DataType dtype,
    int select_dim) {
  auto elem_size = benchmark_state.range(0);
  auto select_size = benchmark_state.range(1);
  int nFeat = 128; // lets fix feat dim for now

  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);

  at::Tensor t0 =
      (select_dim ? at::randn({nFeat, elem_size}, options)
                  : at::randn({elem_size, nFeat}, options));
  auto indx_options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0);
  at::Tensor t1 = at::randint(elem_size, {select_size}, indx_options);

  at::Tensor t2 =
      (select_dim ? at::randn({nFeat, select_size}, options)
                  : at::randn({select_size, nFeat}, options));

  // Sync everything up before we start
  clearL2Cache();
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto output = at::index_select(t0, select_dim, t1) * t2 + 17.0;
    benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    clearL2Cache();
    C10_CUDA_CHECK(cudaDeviceSynchronize());
  }

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (nFeat * select_size * 2 /*2 elemwise ops*/ + select_size +
       nFeat * select_size /*index select op*/) *
      int64_t(dataTypeSize(dtype)));
}

static void Baseline_IndexSelectSimple_Outer_fp32(
    benchmark::State& benchmark_state) {
  Baseline_IndexSelectSimple(benchmark_state, DataType::Float, 0);
}

static void Baseline_IndexSelect_Outer_fp32(benchmark::State& benchmark_state) {
  Baseline_IndexSelect(benchmark_state, DataType::Float, 0);
}

BENCHMARK(Baseline_IndexSelectSimple_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{32768, 32768}, {65536, 65536}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(Baseline_IndexSelect_Outer_fp32)
    // ->RangeMultiplier(2)
    ->Ranges({{128, 32768}, {16, 32768}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
