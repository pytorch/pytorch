#pragma once

#include <torch/csrc/jit/frontend/resolver.h>

namespace torch {
namespace jit {
// Create a Resolver for use in generating LoweredModules for specific backends.
TORCH_API std::shared_ptr<Resolver> loweredModuleResolver();
} // namespace jit
} // namespace torch
