#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API void inlineForkedClosures(std::shared_ptr<Graph>& to_clean);

}
} // namespace jit
} // namespace torch
