#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

// Run TensorExpressions-based fuser.
TORCH_API void fuseTensorExprs(std::shared_ptr<Graph>& graph);

struct TORCH_API RegisterTensorExprFuser
    : public PassManager<RegisterTensorExprFuser> {
  static void registerPass() {
    PassManager::registerPass(fuseTensorExprs);
  }
  static void clearPass() {
    PassManager::clearPass();
  }
};

} // namespace jit
} // namespace torch
