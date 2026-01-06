
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#ifdef USE_CUDA
#endif

namespace torch::jit {

std::function<void(std::shared_ptr<Graph>&)>& getFuseFrozenConvAddReluImpl() {
  static std::function<void(std::shared_ptr<Graph>&)> impl;
  return impl;
}

// Implementation is in frozen_conv_add_relu_fusion.cpp; at runtime the
// implementation is registered in _fuseFrozenConvAddReluImpl. This allows
// the GPU code to be built separately from CPU-only code. If you're
// expecting conv-add-relu fusion to occur but it's not happening, it's
// possible that the GPU code isn't being built or linked properly.
void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph) {
  if (getFuseFrozenConvAddReluImpl()) {
    getFuseFrozenConvAddReluImpl()(graph);
  }
}

} // namespace torch::jit
