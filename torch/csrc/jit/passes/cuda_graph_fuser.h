#pragma once

#include <ATen/Context.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <string>
#include <utility>

namespace torch {
namespace jit {

// Register CudaFuseGraph in custom passes
struct TORCH_API RegisterCudaFuseGraph
    : public PassManager<RegisterCudaFuseGraph> {
  static bool registerPass(bool enabled) {
    TORCH_WARN(
        "RegisterCudaFuseGraph::registerPass() is deprecated. "
        "Please use torch::jit::fuser::cuda::setEnabled().");
    return fuser::cuda::setEnabled(enabled);
  }

  static bool isRegistered() {
    TORCH_WARN(
        "RegisterCudaFuseGraph::isRegistered() is deprecated. "
        "Please use torch::jit::fuser::cuda::isEnabled().");
    return fuser::cuda::isEnabled();
  }
};

struct CudaFuserComparisonCallback {
  using callback_type =
      std::function<void(const Stack&, const Stack&, const std::string&)>;
  bool run_fallback;
  callback_type callback;
};

TORCH_API CudaFuserComparisonCallback getCudaFuserComparisonCallback();
TORCH_API void setCudaFuserComparisonCallback(CudaFuserComparisonCallback);

} // namespace jit
} // namespace torch
