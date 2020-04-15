#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
TORCH_API void insertPrePackedOps(std::shared_ptr<Graph>& graph);
TORCH_API void insertPrePackedOps(script::Module& module);
TORCH_API void fusePrePackedLinearConvWithClamp(script::Module& module);
TORCH_API void FoldPrePackingOps(script::Module& module);
TORCH_API c10::optional<script::Module> optimizeForMobile(const script::Module& module);
} // namespace jit
} // namespace torch
