#pragma once
#include <memory>

#include <torch/csrc/Export.h>

namespace torch::jit {

struct Graph;

// Transforms loops so that they can be represented as python
// for or while loops
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
