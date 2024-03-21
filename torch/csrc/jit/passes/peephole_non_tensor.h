#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// return true if graph is modified
// Optimizing General Graph Patterns that
// are not covered in peephole.cpp and peephole_list_idioms
TORCH_API bool PeepholeOptimizeNonTensor(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
