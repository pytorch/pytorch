#include <ATen/native/MathBitsFallback.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at {

struct NegFallback : MathOpFallback {
  NegFallback() : MathOpFallback(DispatchKey::Negative, "negation") {}
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_neg();
  }
  void _set_bit(const Tensor& tensor, bool value) override {
    return tensor._set_neg(value);
  }
  Tensor resolve_bit(const Tensor& tensor) override {
    return at::resolve_neg(tensor);
  }
  Tensor& math_op_(Tensor& tensor) override {
    return tensor.neg_();
  }
};

void negationFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  NegFallback object;
  object.fallback_impl(op, dispatch_keys, stack);
}

void negationFallbackToHandleOnlyMutableInputs(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  NegFallback object;
  object.linalg_fallback(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, Negative, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&negationFallback>());
}

TORCH_LIBRARY_IMPL(aten, Negative, m) {
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("neg_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_neg", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());

  // linear algebra functions
  m.impl("linalg_solve_triangular", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", torch::CppFunction::makeFromBoxedFunction<&negationFallbackToHandleOnlyMutableInputs>());

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
}

} // namespace at
