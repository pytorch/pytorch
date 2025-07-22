#pragma once

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/C10Kernel.h>

namespace torch::nativert {

#define KernelInput(id) input(id, executionFrame)
#define KernelOutput(id) output(id, executionFrame)

TORCH_DECLARE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*);

#define REGISTER_PRIM_KERNEL(name, id, ...)                          \
  class OpKernel_##id : public OpKernel {                            \
   public:                                                           \
    OpKernel_##id(const Node* node)                                  \
        : OpKernel(node, std::nullopt, OpKernelKind::kPrimKernel) {} \
    void computeInternal(                                            \
        ExecutionFrame& executionFrame) const override final {       \
      __VA_ARGS__;                                                   \
    }                                                                \
  };                                                                 \
  C10_REGISTER_TYPED_CLASS(PrimKernelRegistry, name, OpKernel_##id)

inline bool checkResizedDataPtr(at::Tensor& t) {
  auto const prev_data_ptr = t.data_ptr();
  t.resize_({0});
  return prev_data_ptr == t.data_ptr();
}

inline void fastResizeToZero(at::Tensor& t) {
  t.unsafeGetTensorImpl()->set_sizes_contiguous({0});
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(checkResizedDataPtr(t));
}

} // namespace torch::nativert
