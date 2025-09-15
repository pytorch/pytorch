#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Concats multiple linear ops with the same Tensor input
// into a single linear op.
TORCH_API bool FrozenConcatLinear(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
