#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API std::shared_ptr<Graph> Canonicalize(
    const std::shared_ptr<Graph>& graph,
    bool keep_unique_names = true);

TORCH_API void CanonicalizeOutputs(std::shared_ptr<Graph>& graph);
} // namespace jit
} // namespace torch
