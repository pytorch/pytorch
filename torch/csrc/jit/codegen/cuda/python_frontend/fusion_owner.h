
#pragma once

#include <torch/csrc/jit/codegen/cuda/fusion.h>

using namespace torch::jit::fuser::cuda;

namespace nvfuser {

class FusionOwner {
 public:
  FusionOwner() : executor_cache_(std::make_unique<Fusion>()) {}

  // Non-copyable
  FusionOwner(const FusionOwner&) = delete;
  FusionOwner& operator=(const FusionOwner&) = delete;

  std::vector<at::Tensor> execute(const at::ArrayRef<c10::IValue>& inputs) {
    return executor_cache_.runFusionWithInputs(inputs);
  }
  Fusion* fusionPtr() {
    return executor_cache_.fusion();
  }

  void printIr() {
    executor_cache_.printFusion();
  }
  void printKernel() {
    executor_cache_.fusion()->printKernel();
  }

 private:
  FusionExecutorCache executor_cache_;
};

} // namespace nvfuser
