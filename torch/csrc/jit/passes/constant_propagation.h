#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

TORCH_API void ConstantPropagation(std::shared_ptr<Graph>& graph);
TORCH_API void ConstantPropagation(script::Module& m);
} // namespace jit
} // namespace torch
