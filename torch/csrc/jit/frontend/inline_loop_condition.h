
copy: fbcode/caffe2/torch/csrc/jit/frontend/inline_loop_condition.h
copyrev: f1fb42b05bdd0c74f71bed3fefffafd904578813

#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void InlineLoopCondition(std::shared_ptr<Graph>& graph);
TORCH_API void InlineBlockBeforeNode(Node* before_node, Block* block);

} // namespace jit
} // namespace torch
