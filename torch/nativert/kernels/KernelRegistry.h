#pragma once

#include <ATen/ATen.h>

#include <torch/nativert/executor/OpKernel.h>
#include <torch/nativert/graph/Graph.h>
#include <torch/nativert/kernels/PrimKernelRegistry.h>

namespace torch::nativert {

TORCH_DECLARE_REGISTRY(
    StaticallyDispatchedCPUKernelRegistry,
    OpKernel,
    const Node*);

#define REGISTER_CPU_KERNEL(name, id, ...)                                \
  class OpKernel_##id : public C10Kernel {                                \
   public:                                                                \
    OpKernel_##id(const Node* node)                                       \
        : C10Kernel(                                                      \
              node,                                                       \
              torch::nativert::OpKernelKind::kStaticDispatchKernel) {}    \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame) \
        const override final {                                            \
      __VA_ARGS__;                                                        \
    }                                                                     \
  };                                                                      \
  C10_REGISTER_TYPED_CLASS(                                               \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

#define ALIASING_SPEC(...) __VA_ARGS__

#define REGISTER_ALIASING_CPU_KERNEL(name, id, aliasing_spec, ...)        \
  class OpKernel_##id : public C10Kernel {                                \
   public:                                                                \
    OpKernel_##id(const Node* node)                                       \
        : C10Kernel(                                                      \
              node,                                                       \
              torch::nativert::OpKernelKind::kNativeStaticDispatchKernel, \
              aliasing_spec) {}                                           \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame) \
        const override final {                                            \
      __VA_ARGS__;                                                        \
    }                                                                     \
  };                                                                      \
  C10_REGISTER_TYPED_CLASS(                                               \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

#define REGISTER_NATIVE_CPU_KERNEL(name, id, ...)                            \
  class OpKernel_##id : public C10Kernel {                                   \
   public:                                                                   \
    OpKernel_##id(const Node* node)                                          \
        : C10Kernel(                                                         \
              node,                                                          \
              torch::nativert::OpKernelKind::kNativeStaticDispatchKernel) {} \
    void computeInternal(torch::nativert::ExecutionFrame& executionFrame)    \
        const override final {                                               \
      __VA_ARGS__;                                                           \
    }                                                                        \
  };                                                                         \
  C10_REGISTER_TYPED_CLASS(                                                  \
      StaticallyDispatchedCPUKernelRegistry, name, OpKernel_##id)

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
