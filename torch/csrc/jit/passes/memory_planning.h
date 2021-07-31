#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void planMemory(std::shared_ptr<Graph>&);

} // namespace jit
} // namespace torch
