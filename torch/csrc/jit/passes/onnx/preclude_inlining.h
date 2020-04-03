#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void ONNXFunctionCallSubstitution(Graph& graph);

}
} // namespace torch
