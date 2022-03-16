#pragma once

#include <ATen/Context.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <utility>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
struct TORCH_API RegisterCudaFuseGraph
    : public PassManager<RegisterCudaFuseGraph> {
  static bool registerPass(bool enabled) {
    bool old_flag = PassManager::isRegistered();
    if (enabled) {
      TORCH_CHECK(
          at::globalContext().hasCUDA() && !at::globalContext().hasHIP(),
          "Running CUDA fuser is only supported on CUDA builds.");
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

using CudaFuserComparisonCallback =
    std::function<void(size_t, const Stack&, const Stack&)>;

TORCH_API CudaFuserComparisonCallback getCudaFuserComparisonCallback();
TORCH_API void setCudaFuserComparisonCallback(CudaFuserComparisonCallback);

} // namespace jit
} // namespace torch
