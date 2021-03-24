#pragma once

#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>

#include <cuda_runtime.h>

using namespace torch::jit::fuser::cuda;

static void analyzeFusion(
    Fusion* fusion,
    std::vector<TensorView*>& reduction_tv,
    std::vector<TensorView*>& other_tv) {
  auto all_values = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(all_values)) {
    if (tv->hasReduction()) {
      reduction_tv.push_back(tv);
    } else if (!fusion->hasInput(tv)) {
      other_tv.push_back(tv);
    }
  }
}

class CudaKernelTimer {
 public:
  CudaKernelTimer() {
    // Setup
    cudaEventCreate(&start_event);
    cudaEventCreate(&finish_event);
    cudaEventRecord(start_event);
  }

  ~CudaKernelTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(finish_event);
  }

  float elapsed() {
    // Record
    cudaEventRecord(finish_event);
    cudaEventSynchronize(start_event);
    cudaEventSynchronize(finish_event);
    cudaEventElapsedTime(&kernel_time_ms_, start_event, finish_event);
    return kernel_time_ms_;
  }

 private:
  // Create
  float kernel_time_ms_ = 0;
  cudaEvent_t start_event = {};
  cudaEvent_t finish_event = {};
};
