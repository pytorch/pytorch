#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch::jit {
struct Graph;

// Propagate tensor properties (e.g., dtype, device, is_contiguous, layout)
// propagation on all tensor objects. Currently, we only support dtype
// propagation
TORCH_API bool DtypePropagation(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
