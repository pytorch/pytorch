#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void FixupONNXConditionals(std::shared_ptr<Graph>& graph);

}
} // namespace torch
