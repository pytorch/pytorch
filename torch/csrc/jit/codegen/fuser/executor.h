#pragma once

#include <ATen/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>
#include <torch/csrc/jit/codegen/fuser/kernel_spec.h>

#include <cstdint>

namespace torch {
namespace jit {
namespace fuser {

// Runs the fusion associated with the key (see registerFusion() in interface.h)
// on the inputs taken from the given Stack.
TORCH_API bool runFusion(
    const int64_t key,
    Stack& stack,
    std::string* code_out = nullptr);

} // namespace fuser
} // namespace jit
} // namespace torch
