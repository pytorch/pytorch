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

#include "utils.h"

using namespace torch::jit::fuser::cuda;

// Return reduction tensor view and output of reduction
static std::pair<TensorView*, TensorView*> setupReduction(
    Fusion* fusion,
    DataType dtype,
    int red_axis) {

  FusionGuard fg(fusion);

  bool is_fp16 = dtype == DataType::Half;

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(2)
                        .dtype(dtype)
                        .contiguity({true, true})
                        .build();
  fusion->addInput(tv0);

  TensorView* tv0_cast = tv0;
  if (is_fp16) {
    tv0_cast = castOp(DataType::Float, tv0);
  }

  TensorView* tv1 = sum(tv0_cast, {red_axis});

  TensorView* tv1_cast = tv1;
  if (is_fp16) {
    tv1_cast = castOp(DataType::Half, tv1);
  }

  fusion->addOutput(tv1_cast);

  TensorView* output_of_reduction = nullptr;
  if (is_fp16) {
    output_of_reduction = tv1_cast;
  }

  return {tv1, output_of_reduction};
}

static void MagicScheduler_Reduction(benchmark::State& benchmark_state,
  DataType dtype,
  int reduction_dim) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  auto reduction_size = benchmark_state.range(0);
  auto iter_size = benchmark_state.range(1);

  auto reduction_tvs = setupReduction(&fusion, dtype, reduction_dim);

  at::manual_seed(0);
  auto options = at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor aten_input =
      (reduction_dim ? at::randn({iter_size, reduction_size}, options)
            : at::randn({reduction_size, iter_size}, options));

  auto reduction_tv = reduction_tvs.first;
  auto out_of_reduction = reduction_tvs.second;

  auto reduction_params =
      getReductionHeuristics(&fusion, {aten_input}, reduction_tv);

  TORCH_CHECK(reduction_params.has_value(), "Reduction is not found!");
  
  std::vector<TensorView*> outputs_of_reduction;
  if(out_of_reduction != nullptr){
    outputs_of_reduction.push_back(out_of_reduction);
  }

  auto rparams = reduction_params.value();
  auto lparams = rparams.lparams;

  scheduleReduction(
      &fusion, rparams, reduction_tv, outputs_of_reduction);

  std::stringstream ss;
  if(rparams.fastest_dim){
    ss << "Fastest dim";
  } else {
    ss << "Slow dim";
  }
  if(rparams.cross_block){
    ss << "/cross block";
  }
  if(rparams.multiple_reds_per_blk){
    ss << "/multiple reductions per block ";
  }
  if(rparams.cross_grid){
    ss << "/cross grid";
  }
  if(rparams.loop_unroll > 1){
    ss << "/Unroll "
       << (rparams.reduction_unroll ? "reduction dim "
                                                     : "iter dim ")
       << rparams.loop_unroll;
  }
  ss << "/Launch (" << (rparams.fastest_dim ? lparams.gdimx() : lparams.gdimy())
     << ", " << lparams.bdimy() << ", " << lparams.bdimx() << ")";

  benchmark_state.SetLabel(ss.str());
  

  FusionExecutor fe;
  fe.compileFusion(&fusion);
  fe.setMeasureKernelTimeFlag(true);
  // Sync everything up before we start
  cudaDeviceSynchronize();
  for (auto _ : benchmark_state) {
    CudaKernelTimer timer;
    auto cg_outputs = fe.runFusion({aten_input}, lparams);
    benchmark_state.SetIterationTime(fe.kernelTimeMs() / 1000.0);
  }
  // Sync everything up before we're finished, don't want to run ahead on the
  // cpu while benchmarking.
  cudaDeviceSynchronize();

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (iter_size * reduction_size + iter_size) * int64_t(dataTypeSize(dtype)));
}

static void MagicScheduler_fp32_Outer_Reduction(benchmark::State& benchmark_state) {
  MagicScheduler_Reduction(benchmark_state, DataType::Float, 0);
}

static void MagicScheduler_fp32_Inner_Reduction(benchmark::State& benchmark_state) {
  MagicScheduler_Reduction(benchmark_state, DataType::Float, 1);
}

static void MagicScheduler_fp16_Outer_Reduction(benchmark::State& benchmark_state) {
  MagicScheduler_Reduction(benchmark_state, DataType::Half, 0);
}

static void MagicScheduler_fp16_Inner_Reduction(benchmark::State& benchmark_state) {
  MagicScheduler_Reduction(benchmark_state, DataType::Half, 1);
}

BENCHMARK(MagicScheduler_fp32_Outer_Reduction)
    ->RangeMultiplier(8)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp32_Outer_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{32768, 128 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp32_Outer_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{2, 16}, {32768, 128 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Outer_Reduction)
    ->RangeMultiplier(8)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Outer_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{32768, 128 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Outer_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{2, 16}, {32768, 128 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp32_Inner_Reduction)
    ->RangeMultiplier(8)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp32_Inner_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{32768, 128 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp32_Inner_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{2, 16}, {32768, 128 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Inner_Reduction)
    ->RangeMultiplier(8)
    ->Ranges({{1, 1024 * 1024}, {160, 320}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Inner_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{32768, 128 * 1024 * 1024}, {2, 16}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

BENCHMARK(MagicScheduler_fp16_Inner_Reduction)
    ->RangeMultiplier(4)
    ->Ranges({{2, 16}, {32768, 128 * 1024 * 1024}})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
