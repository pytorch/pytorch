#pragma once

#include <ATen/Utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace at {
namespace native {
at::Tensor& reshape_copy_out(
    at::Tensor& out,
    const at::Tensor& self,
    const std::vector<int64_t>& proposed_shape,
    bool infer_size = true);
} // namespace native
} // namespace at

namespace torch {
namespace jit {

using SROperator = std::function<void(ProcessedNode*)>;
using SROpFunctor = SROperator (*)(Node* n);
struct SROperatorFunctor {
  virtual SROperator Generate(Node*) {
    std::function<void(ProcessedNode*)> out;
    return out;
  }
  virtual ~SROperatorFunctor() = default;
};

C10_DECLARE_REGISTRY(SROperatorRegistry, SROperatorFunctor);

#define REGISTER_OPERATOR_FUNCTOR(name, id, ...)             \
  struct SROperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                      \
    SROperator Generate(Node* n) override {                  \
      return fn(n);                                          \
    }                                                        \
  };                                                         \
  C10_REGISTER_CLASS(SROperatorRegistry, name, SROperatorFunctor_##id);

C10_DECLARE_REGISTRY(SRNativeOperatorRegistry, SROperatorFunctor);
#define REGISTER_NATIVE_OPERATOR_FUNCTOR(name, id, ...)            \
  struct SRNativeOperatorFunctor_##id : public SROperatorFunctor { \
    const SROpFunctor fn = __VA_ARGS__;                            \
    SROperator Generate(Node* n) override {                        \
      return fn(n);                                                \
    }                                                              \
  };                                                               \
  C10_REGISTER_CLASS(                                              \
      SRNativeOperatorRegistry, name, SRNativeOperatorFunctor_##id);

inline at::Tensor create_empty_from(const at::Tensor& t) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      c10::nullopt,
      c10::nullopt);
}

inline at::Tensor create_empty_from(
    at::IntArrayRef sizes,
    const at::Tensor& t) {
  return at::detail::empty_cpu(
      sizes,
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      c10::nullopt,
      c10::nullopt);
}

inline at::Tensor create_empty(c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), c10::nullopt, c10::nullopt);
}

inline at::Tensor create_empty_from(const at::Tensor& t, c10::Layout layout) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      layout,
      t.device(),
      c10::nullopt,
      c10::nullopt);
}

inline at::Tensor create_empty_from(const at::Tensor& t, c10::Device device) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      device,
      c10::nullopt,
      c10::nullopt);
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::MemoryFormat memory_format) {
  return at::detail::empty_cpu(
      {0},
      c10::typeMetaToScalarType(t.dtype()),
      t.layout(),
      t.device(),
      c10::nullopt,
      memory_format);
}

inline bool checkResizedDataPtr(at::Tensor& t) {
  auto const prev_data_ptr = t.data_ptr();
  t.resize_({0});
  return prev_data_ptr == t.data_ptr();
}

inline void fastResizeToZero(at::Tensor& t) {
  t.unsafeGetTensorImpl()->set_sizes_contiguous({0});
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(checkResizedDataPtr(t));
}

// check if an op has an out variant registered in Static Runtime
bool opIsRegistered(const c10::Symbol& op_name);
// check if Static Runtime can run an op natively.
// prim ops that are implemented directly in the jit interpreter are implemented
// as native ops in Static Runtime
bool nativeOpIsRegistered(const c10::Symbol& op_name);

bool canReuseInputsOutputs(Node* n);
bool isOptimizableContainerType(Node* n);

std::function<void(ProcessedNode*)> getOutOfPlaceOperation(Node* n);
std::function<void(ProcessedNode*)> getNativeOperation(Node* n);

inline std::string PrintNode(const Node* node) {
  std::ostringstream ss;
  node->print(ss, 0, nullptr, false);
  return ss.str();
}

inline void LogAndDumpSchema(const Node* node) {
  VLOG(1) << "Found schema mismatch";
  node->schema().dump();
}

} // namespace jit
} // namespace torch
