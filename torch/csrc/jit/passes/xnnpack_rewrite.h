#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

enum class MobileOptimizerType : int8_t {
  CONV_BN_FUSION,
  INSERT_FOLD_PREPACK_OPS,
  REMOVE_DROPOUT,
  FUSE_ADD_RELU,
  HOIST_CONV_PACKED_PARAMS,
};

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
