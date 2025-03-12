#pragma once

#include <torch/script.h>

#include "torch/csrc/nativert/executor/OpKernel.h"
#include "torch/csrc/nativert/graph/Graph.h"
#include "torch/csrc/nativert/kernels/C10Kernel.h"

namespace torch::nativert {

#define KernelInput(id) input(id, executionFrame)
#define KernelOutput(id) output(id, executionFrame)

TORCH_DECLARE_REGISTRY(PrimKernelRegistry, OpKernel, const Node*);

#define REGISTER_PRIM_KERNEL(name, id, ...)                    \
  class OpKernel_##id : public OpKernel {                      \
   public:                                                     \
    OpKernel_##id(const Node* node) : OpKernel(node) {         \
      kind_ = OpKernel::Kind::kPrimKernel;                     \
    }                                                          \
    void computeInternal(                                      \
        ExecutionFrame& executionFrame) const override final { \
      __VA_ARGS__;                                             \
    }                                                          \
  };                                                           \
  C10_REGISTER_TYPED_CLASS(PrimKernelRegistry, name, OpKernel_##id);

TORCH_DECLARE_REGISTRY(
    StaticallyDispatchedCPUKernelRegistry,
    OpKernel,
    const Node*,
    c10::Device);

#define REGISTER_CPU_KERNEL(name, id, ...)                     \
  class OpKernel_##id : public C10Kernel {                     \
   public:                                                     \
    OpKernel_##id(const Node* node, c10::Device device)        \
        : C10Kernel(node, device) {                            \
      kind_ = OpKernel::Kind::kStaticDispatchKernel;           \
    }                                                          \
    void computeInternal(                                      \
        ExecutionFrame& executionFrame) const override final { \
      __VA_ARGS__;                                             \
    }                                                          \
  };                                                           \
  C10_REGISTER_TYPED_CLASS(                                    \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id);

inline bool checkResizedDataPtr(at::Tensor& t) {
  auto const prev_data_ptr = t.data_ptr();
  t.resize_({0});
  return prev_data_ptr == t.data_ptr();
}

inline void fastResizeToZero(at::Tensor& t) {
  t.unsafeGetTensorImpl()->set_sizes_contiguous({0});
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(checkResizedDataPtr(t));
}

inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      std::nullopt,
      std::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), std::nullopt, std::nullopt);
}

inline at::Tensor create_empty_from(const at::Tensor& t, c10::Device device) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      device,
      std::nullopt,
      std::nullopt);
}
inline at::Tensor create_empty_from(const at::Tensor& t, c10::Layout layout) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      layout,
      t.device(),
      std::nullopt,
      std::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::MemoryFormat memory_format) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      std::nullopt,
      memory_format);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype,
    c10::MemoryFormat memory_format) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), std::nullopt, memory_format);
}

} // namespace torch::nativert
