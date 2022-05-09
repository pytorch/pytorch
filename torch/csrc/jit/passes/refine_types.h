#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// updates the types of container data structures (e.g. tuple) according to
// the type of their current inputs.
TORCH_API void RefineTypes(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
