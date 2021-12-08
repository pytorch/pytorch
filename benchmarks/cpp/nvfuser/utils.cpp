#include "utils.h"

#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <sstream>

using namespace torch::jit::fuser::cuda;

std::string toString(ReductionParams rparams) {
  std::stringstream ss;
  ss << (rparams.fastest_dim ? "Red On Fastest Dim // " : "Red On Slow Dim // ")
     << (rparams.persistent_kernel ? "Persistent Kernel // " : "")
     << (rparams.project_persistent_buffers ? "Project Persistent Buffers // "
                                            : "");

  if (rparams.schedule_3D) {
    ss << "3D Schedule // "
       << "Outer Reduction: "
       << (rparams.cross_block_outer_reduction ? "cross block / " : "")
       << (rparams.cross_grid_outer_reduction ? "cross grid / " : "")
       << (rparams.split_grid_dim_outer_reduction ? "split grid dim / " : "");
    if (rparams.batches_per_block_outer_reduction > 1 ||
        rparams.persistent_kernel) {
      ss << "persistent batch - " << rparams.batches_per_block_outer_reduction
         << " / ";
    }
  }

  ss << " // Iteration Domain: "
     << (rparams.multiple_reds_per_blk ? "multiple reductions per block / "
                                       : "")
     << (rparams.split_grid_dim_iter_dom ? "split grid dimension / " : "")
     << (rparams.vectorize_iter_dom ? "vectorize / " : "")
     << (rparams.unroll_iter_dom && !rparams.vectorize_iter_dom ? "unroll / "
                                                                : "");
  if (rparams.unroll_iter_dom || rparams.vectorize_iter_dom) {
    ss << "factor " << rparams.unroll_factor_iter_dom;
  }

  ss << " // Inner Reduction Domain: "
     << (rparams.cross_block_inner_reduction ? "cross block reduction / " : "")
     << (rparams.pad_inner_reduction_to_warp ? "pad to warp / " : "")
     << (rparams.cross_grid_inner_reduction ? "cross grid reduction / " : "");

  if (rparams.batches_per_block_inner_reduction > 1 ||
      rparams.persistent_kernel) {
    ss << "persistent batch - " << rparams.batches_per_block_inner_reduction
       << " / ";
  }

  ss << (rparams.cross_grid_inner_reduction &&
                 rparams.split_grid_dim_inner_reduction
             ? "split grid dimension / "
             : "")
     << (rparams.vectorize_inner_reduction ? "vectorize / " : "")
     << (rparams.unroll_inner_reduction && !rparams.vectorize_inner_reduction
             ? "unroll / "
             : "");
  if (rparams.unroll_inner_reduction || rparams.vectorize_inner_reduction) {
    ss << "factor " << rparams.unroll_factor_inner_reduction;
  }
  return ss.str();
}

std::string toString(PointwiseParams params) {
  std::stringstream ss;
  if (params.break_point) {
    ss << "2D Schedule at " << params.break_point << "/";
    if (params.split_block) {
      ss << " Split block into y-dim/";
    }
    if (params.split_grid_y_dim) {
      ss << " Split y grid dim/";
    }
  } else {
    ss << "1D"
       << "/";
  }
  if (params.inner_factor > 1) {
    if (params.vectorize) {
      ss << "Vectorize, Factor: " << params.inner_factor;
    } else {
      ss << "Unroll, Factor: " << params.inner_factor;
    }
  }
  return ss.str();
}

std::string toString(LaunchParams lparams) {
  std::stringstream ss;
  lparams.toString();
  ss << "/Launch_Parameters["
     << "block(" << lparams.bdimz() << "/" << lparams.bdimy() << "/"
     << lparams.bdimx() << ")/grid(" << lparams.gdimz() << "/"
     << lparams.gdimy() << "/" << lparams.gdimx() << ")/" << lparams.smem()
     << "]";
  return ss.str();
}

void clearL2Cache() {
  torch::NoGradGuard no_grad;
  auto l2_cache_size = at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(at::kCUDA, 0);

  auto l2_elems = l2_cache_size / 4;
  torch::Tensor t0 = torch::empty(l2_elems, options);
  torch::Tensor t1 = torch::clone(t0);
};

TensorView* makeContigTensor(size_t ndims, DataType dtype) {
  return TensorViewBuilder()
      .ndims(ndims)
      .dtype(dtype)
      .contiguity(std::vector<bool>(ndims, true))
      .build();
}

void runBenchmarkIterations(
    benchmark::State& benchmark_state,
    FusionExecutorCache* fusion_executor_cache,
    std::vector<c10::IValue>& aten_inputs) {
  fusion_executor_cache->runFusionWithInputs(aten_inputs);
  bool segmented =
      fusion_executor_cache->getMostRecentKernelRuntime()->isSegmented();

  if (!segmented) {
    fusion_executor_cache->profile(true);
    fusion_executor_cache->runFusionWithInputs(aten_inputs);
    auto compile_log = fusion_executor_cache->getMostRecentExecutorInfo();
    auto executor_instance = compile_log.fusion_executor;
    TORCH_INTERNAL_ASSERT(compile_log.reduction_params.has_value());
    TORCH_INTERNAL_ASSERT(compile_log.launch_constraints.has_value());
    auto rparams = toString(compile_log.reduction_params.value());
    auto lparams = toString(compile_log.launch_constraints.value());
    benchmark_state.SetLabel(rparams + lparams);
    executor_instance->setMeasureKernelTimeFlag(true);

    // Sync everything up before we start
    cudaDeviceSynchronize();
    for (auto _ : benchmark_state) {
      clearL2Cache();
      auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
      benchmark_state.SetIterationTime(
          executor_instance->kernelTimeMs() / 1000.0);
    }
    // Sync everything up before we're finished, don't want to run ahead on the
    // cpu while benchmarking.
    cudaDeviceSynchronize();
  } else {
    // Segmented
    // Sync everything up before we start
    {
      // Compile/warmup
      auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
    }
    cudaDeviceSynchronize();
    CudaKernelTimer timer;
    for (auto _ : benchmark_state) {
      clearL2Cache();
      timer.restart();
      auto cg_outputs = fusion_executor_cache->runFusionWithInputs(aten_inputs);
      benchmark_state.SetIterationTime(timer.elapsed() / 1000.0);
    }
    // Sync everything up before we're finished, don't want to run ahead on the
    // cpu while benchmarking.
    cudaDeviceSynchronize();
  }
}

namespace executorCache {
thread_local ExecutorMap executor_map_;
ExecutorMap& getGlobalMap() {
  return executor_map_;
}
} // namespace executorCache
