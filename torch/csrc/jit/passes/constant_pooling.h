#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void ConstantPooling(const std::shared_ptr<Graph>& graph);

TORCH_API void ConstantPoolingImmutableTypes(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
