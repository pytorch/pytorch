#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// EliminateUnusedItemsONNX pass is removing unused
// initializers and inputs, this is needed because
// dce pass is only removing unused fork inputs
void EliminateUnusedItemsONNX(
    Block* b,
    std::map<std::string, IValue>& paramDict);

} // namespace torch::jit
