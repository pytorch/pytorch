#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

TORCH_API void FixupTraceScopeBlocks(
    std::shared_ptr<Graph>& graph,
    script::Module* self);

} // namespace jit
} // namespace torch
