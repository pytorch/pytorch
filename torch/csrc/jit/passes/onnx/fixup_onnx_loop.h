#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FixupONNXLoops(std::shared_ptr<Graph>& graph);

}
} // namespace torch
