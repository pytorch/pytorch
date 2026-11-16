#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

void PreprocessForONNX(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
