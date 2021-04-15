#pragma once

#include <ATen/Context.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
struct C10_EXPORT RegisterCudaFuseGraph
    : public PassManager<RegisterCudaFuseGraph> {
  static bool registerPass(bool enabled) {
    TORCH_CHECK(
        at::globalContext().hasCUDA() && !at::globalContext().hasHIP(),
        "Running CUDA fuser is only supported on CUDA builds.");
    bool old_flag = PassManager::isRegistered();
    if (enabled) {
      PassManager::registerPass(fuser::cuda::fuseGraph);
    } else {
      PassManager::clearPass();
    }
    return old_flag;
  }

  static bool isRegistered() {
    return PassManager::isRegistered();
  }
};

} // namespace jit
} // namespace torch
