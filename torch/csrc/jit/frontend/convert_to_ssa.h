#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Convert a graph with Loads & Stores into SSA form
TORCH_API void ConvertToSSA(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
