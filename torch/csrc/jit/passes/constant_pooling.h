#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

TORCH_API void ConstantPooling(const std::shared_ptr<Graph>& graph);

}
} // namespace torch
