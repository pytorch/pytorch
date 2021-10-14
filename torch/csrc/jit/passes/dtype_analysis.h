#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Propagate tensor properties (e.g., dtype, device, is_contiguous, layout)
// propagation on all tensor objects. Currently, we only support dtype
// propagation
TORCH_API bool DtypePropagation(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
