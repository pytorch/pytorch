#pragma once

#include <ATen/ATen.h>
#include <utility>

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/C10Kernel.h>

namespace torch::nativert {

#define KernelInput(id) input(id, executionFrame)
// When memory planning is enabled, LayoutManager already handles resize-to-zero
// (see LayoutManager.cpp ensure_managed_storages), so we skip fastResizeToZero.
// Always set _outputResized[idx] = true on ANY access (not just when iv is a
// tensor) to handle kernels that first check isNone(), assign, then access
// again. Without this, the second access would incorrectly call
// fastResizeToZero.
#define KernelOutput(idx)                                             \
  ([&]() -> c10::IValue& {                                            \
    if (executionFrame.isManagedValue(node_->outputs()[idx]->id())) { \
      return output(idx, executionFrame);                             \
    }                                                                 \
    c10::IValue& iv = output(idx, executionFrame);                    \
    /* std::exchange doesn't work with std::vector<bool> */           \
    bool old = _outputResized[idx];                                   \
    _outputResized[idx] = true;                                       \
    if (iv.isTensor() && !old) {                                      \
      fastResizeToZero(iv.toTensor());                                \
    }                                                                 \
    return iv;                                                        \
  }())

// Bypass auto-resize for special cases (e.g., _to_copy with aliasing)
#define KernelOutputUnsafe(id) output(id, executionFrame)

TORCH_DECLARE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*);

#define REGISTER_PRIM_KERNEL(name, id, ...)                    \
  class OpKernel_##id : public OpKernel {                      \
   public:                                                     \
    OpKernel_##id(const Node* node)                            \
        : OpKernel(node, OpKernelKind::kPrimKernel) {}         \
    void computeInternal(                                      \
        ExecutionFrame& executionFrame) const override final { \
      __VA_ARGS__;                                             \
    }                                                          \
  };                                                           \
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
