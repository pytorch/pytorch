#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Inline Fork and Wait calls. This is used, for example, in ONNX export, where
// we do not support the explicit parallelism structures and would rather
// just have a flat graph. This inlines the forked section in the fork()
// callsite and replaces uses of the result of wait() calls with the values
// produced from the (now-inlined) forked section.
TORCH_API void InlineForkWait(const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
