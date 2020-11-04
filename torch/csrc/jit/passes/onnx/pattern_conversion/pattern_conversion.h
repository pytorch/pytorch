#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API std::vector<Value*> ConvertPatternFromSubblock(
    Block* new_block,
    Node* old_node,
    std::unordered_map<Value*, Value*>& env);

} // namespace jit
} // namespace torch
