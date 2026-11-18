#pragma once

#include <memory>

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

void EvalPeepholeONNX(
    std::shared_ptr<Graph>& g,
    std::map<std::string, IValue>& paramDict);

} // namespace torch::jit
