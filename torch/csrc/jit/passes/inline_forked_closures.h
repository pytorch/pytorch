#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API void inlineForkedClosures(std::shared_ptr<Graph>& to_clean);

} // namespace jit
} // namespace torch
