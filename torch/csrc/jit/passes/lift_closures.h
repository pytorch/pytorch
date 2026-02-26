#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

TORCH_API void liftClosures(const std::shared_ptr<Graph>& graph);

} // namespace torch::jit
