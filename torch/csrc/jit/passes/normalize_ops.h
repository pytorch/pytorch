#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// This pass converts aten ops to a normalized form. It is
// run immediately after IR generation in both the tracer and compiler,
// so downstream consumers of the IR do not need handle ops in their
// pre-normalized form.
// Currently only handles normalization of op aliases.
TORCH_API void NormalizeOps(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
