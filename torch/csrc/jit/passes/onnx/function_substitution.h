#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void ONNXFunctionCallSubstitution(Graph& graph);

}
} // namespace torch
