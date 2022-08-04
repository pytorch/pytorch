
#pragma once

#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>

using NvfFusionExecutorCache = torch::jit::fuser::cuda::FusionExecutorCache;
using NvfFusion = torch::jit::fuser::cuda::Fusion;

namespace nvfuser {

class FusionOwner {
 public:
  FusionOwner() : executor_cache_(std::make_unique<NvfFusion>()) {}

  // Non-copyable
  FusionOwner(const FusionOwner&) = delete;
  FusionOwner& operator=(const FusionOwner&) = delete;

  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs) {
    return executor_cache_.runFusionWithInputs(inputs);
  }
  NvfFusion* fusionPtr() {
    return executor_cache_.fusion();
  }

  void printIr() {
    executor_cache_.printFusion();
  }
  void printKernel() {
    executor_cache_.fusion()->printKernel();
  }

 private:
  NvfFusionExecutorCache executor_cache_;
};

} // namespace nvfuser
