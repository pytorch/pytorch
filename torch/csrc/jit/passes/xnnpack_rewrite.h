#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>

namespace torch {
namespace jit {

TORCH_API void transformConv1dToConv2d(std::shared_ptr<Graph>& graph);
TORCH_API void transformConv1dToConv2d(script::Module& module);
TORCH_API void insertPrePackedOps(std::shared_ptr<Graph>& graph);
TORCH_API void insertPrePackedOps(script::Module& module);
TORCH_API void fusePrePackedLinearConvWithClamp(script::Module& module);
TORCH_API void FoldPrePackingOps(script::Module& module);
TORCH_API script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& optimization_blocklist = {},
    const std::vector<std::string>& preserved_methods = {});
} // namespace jit
} // namespace torch
