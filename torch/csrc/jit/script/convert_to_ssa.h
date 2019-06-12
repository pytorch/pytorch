#pragma once
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

// Convert a graph with Loads & Stores into SSA form
TORCH_API void ConvertToSSA(std::shared_ptr<Graph>& graph);

} // namespace script
} // namespace jit
} // namespace torch
