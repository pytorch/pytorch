#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <memory> // for shared_ptr

#include "c10/macros/Export.h" // for TORCH_API

namespace torch {
namespace jit {
struct Graph;

// Propagate tensor properties (e.g., dtype, device, is_contiguous, layout)
// propagation on all tensor objects. Currently, we only support dtype
// propagation
TORCH_API bool DtypePropagation(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
