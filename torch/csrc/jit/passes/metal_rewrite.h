#pragma once
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <string>
#include <vector>

namespace torch::jit {
TORCH_API void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph);
TORCH_API void metalInsertPrePackedOps(script::Module& module);
TORCH_API void metalFusePrePackedConvWithClamp(script::Module& module);
TORCH_API void metalFoldPrePackingOps(script::Module& module);
TORCH_API script::Module metalOptimizeForMobile(
    const script::Module& module,
    const std::vector<std::string>& preserved_methods);
} // namespace torch::jit
