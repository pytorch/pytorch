#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
TORCH_API void vulkanInsertPrePackedOps(std::shared_ptr<Graph>& graph);
TORCH_API void vulkanInsertPrePackedOps(script::Module& module);
TORCH_API void vulkanFusePrePackedConvWithClamp(script::Module& module);
TORCH_API void vulkanFoldPrePackingOps(script::Module& module);
TORCH_API script::Module vulkanOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods);
} // namespace jit
} // namespace torch
