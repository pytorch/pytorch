#pragma once

#include <torch/csrc/api/include/torch/jit.h>
#include <torch/csrc/lazy/backend/lowering_context.h>

namespace torch {
namespace lazy {
using TSOpVector = std::vector<torch::jit::Value*>;

TORCH_API TSOpVector LowerTSBuiltin(
    std::shared_ptr<torch::jit::GraphFunction> function,
    c10::Symbol sym,
    const std::vector<torch::jit::NamedValue>& arguments,
    const std::vector<torch::jit::NamedValue>& kwarguments = {});

} // namespace lazy
} // namespace torch
