#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

enum class MobileOptimizerType {
  FOLD_CONV_BATCH_NORM,
  INSERT_FOLD_PREPACK_OPS
};

TORCH_API void insertPrePackedOps(std::shared_ptr<Graph>& graph);
TORCH_API void insertPrePackedOps(script::Module& module);
TORCH_API void fusePrePackedLinearConvWithClamp(script::Module& module);
TORCH_API void FoldPrePackingOps(script::Module& module);
TORCH_API script::Module optimizeForMobile(
    const script::Module& module,
    const std::map<MobileOptimizerType, bool>& whitelist_optimizers = {});
} // namespace jit
} // namespace torch
