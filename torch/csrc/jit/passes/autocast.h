
#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void Autocast(const std::shared_ptr<Graph>& graph);

TORCH_API bool setAutocastMode(bool value);
TORCH_API bool autocastEnabled();

} // namespace jit
} // namespace torch
