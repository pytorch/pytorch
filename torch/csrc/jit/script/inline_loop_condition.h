#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API void InlineLoopCondition(std::shared_ptr<Graph>& graph);
TORCH_API void inlineBlockBeforeNode(Node* before_node, Block* block);

} // namespace script
} // namespace jit
} // namespace torch
