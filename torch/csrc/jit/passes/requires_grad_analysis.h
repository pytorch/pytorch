#pragma once

#include <torch/csrc/Export.h>

#include <memory>

namespace torch::jit {

struct Graph;
struct ArgumentSpec;

TORCH_API void PropagateRequiresGrad(std::shared_ptr<Graph>& graph);

} // namespace torch::jit
