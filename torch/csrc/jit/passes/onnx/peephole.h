#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

void PeepholeOptimizeONNX(std::shared_ptr<Graph>& graph, int opset_version, bool fixed_batch_size);

}
} // namespace torch
