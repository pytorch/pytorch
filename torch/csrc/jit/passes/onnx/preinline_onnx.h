#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void PreInlineONNX(Graph& graph);

}
} // namespace torch
