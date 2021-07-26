#include <ATen/native/MathBitsFallback.h>

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

TORCH_LIBRARY_IMPL(_, Negative, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&negationFallback>());
}

TORCH_LIBRARY_IMPL(aten, Negative, m) {
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj", torch::CppFunction::makeFallthrough());
  m.impl("neg_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_neg", torch::CppFunction::makeFallthrough());
  m.impl("empty_like", torch::CppFunction::makeFallthrough());
  m.impl("empty.memory_format", torch::CppFunction::makeFallthrough());
  m.impl("empty.out", torch::CppFunction::makeFallthrough());
  m.impl("empty_strided", torch::CppFunction::makeFallthrough());
  m.impl("full_like", torch::CppFunction::makeFallthrough());
  m.impl("stride.int", torch::CppFunction::makeFallthrough());
  m.impl("stride.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("size.int", torch::CppFunction::makeFallthrough());
  m.impl("size.Dimname", torch::CppFunction::makeFallthrough());
  m.impl("is_complex", torch::CppFunction::makeFallthrough());
  m.impl("is_floating_point", torch::CppFunction::makeFallthrough());
  m.impl("view_as_real", torch::CppFunction::makeFallthrough());
  m.impl("imag", torch::CppFunction::makeFallthrough());
  m.impl("real", torch::CppFunction::makeFallthrough());
  m.impl("view", torch::CppFunction::makeFallthrough());
  m.impl("_unsafe_view", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
}

} // namespace at
