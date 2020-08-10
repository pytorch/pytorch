#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

// Run TensorExpressions-based fuser.
TORCH_API void FuseTensorExprs(std::shared_ptr<Graph>& graph);

TORCH_API void setTensorExprFuserEnabled(bool val);
TORCH_API bool tensorExprFuserEnabled();

namespace tensorexpr {
TORCH_API bool isSupported(Node* node);
}
} // namespace jit
} // namespace torch
