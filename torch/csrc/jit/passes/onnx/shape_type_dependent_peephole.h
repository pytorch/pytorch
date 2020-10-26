#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch { namespace jit {

TORCH_API void ShapeTypeDependentPreprocess(const std::shared_ptr<Graph>& graph);
TORCH_API std::vector<Value*> ShapeTypeDependentPeephole(Block* new_block, Node* old_node, std::unordered_map<Value*, Value*>& env);
// TORCH_API void ShapeTypeDependentPeephole(PeepholeContext& ctx);

}} // namespace torch::jit