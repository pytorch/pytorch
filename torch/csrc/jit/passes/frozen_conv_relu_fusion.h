#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void FuseFrozenConvRelu(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
