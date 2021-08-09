#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void planMemory(std::shared_ptr<Graph>&);
TORCH_API std::map<Value*, std::vector<int>> computeLiveness(
    std::shared_ptr<Graph>& graph);
TORCH_API size_t computeStorageSize(const c10::TensorTypePtr& ttp);
TORCH_API std::vector<Node*> findOutVariantNodes(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
