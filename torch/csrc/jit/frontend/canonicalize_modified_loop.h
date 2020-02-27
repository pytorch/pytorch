
copy: fbcode/caffe2/torch/csrc/jit/frontend/canonicalize_modified_loop.h
copyrev: 4d92033813960ebaaf4a564f4a0aa30c67e95e19

#pragma once
#include <memory>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {

struct Graph;

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
