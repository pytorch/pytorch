#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void insertIdentityForInputUsedAsOutput(Block* b);

void PeepholeOptimizeONNX(
    std::shared_ptr<Graph>& graph,
    int opset_version,
    bool fixed_batch_size);

}
} // namespace torch
