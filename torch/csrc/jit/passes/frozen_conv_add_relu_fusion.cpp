#include <ATen/Utils.h>

#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>
#endif

namespace torch {
namespace jit {

std::function<void(std::shared_ptr<Graph>&)>& getFuseFrozenConvAddReluImpl() {
  static std::function<void(std::shared_ptr<Graph>&)> impl;
  return impl;
}

// Implementation is in frozen_conv_add_relu_fusion.cpp; at runtime the
// implementation is registered in _fuseFrozenConvAddReluImpl. This allows
// the GPU code to be built separately from CPU-only code.
void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph) {
  if (getFuseFrozenConvAddReluImpl()) {
    getFuseFrozenConvAddReluImpl()(graph);
  } else {
    TORCH_WARN("No definition of _fuseFrozenConvAddReluImpl found");
  }
}

} // namespace jit
} // namespace torch
