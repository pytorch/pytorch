#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/stack.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

// Runs the fusion associated with the key (see registerFusion() in interface.h)
// on the inputs taken from the given Stack.
TORCH_API bool runFusion(const int64_t key, Stack& stack);

} // namespace fuser
} // namespace jit
} // namespace torch
