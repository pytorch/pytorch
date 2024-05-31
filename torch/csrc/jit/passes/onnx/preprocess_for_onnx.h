#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void PreprocessForONNX(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
