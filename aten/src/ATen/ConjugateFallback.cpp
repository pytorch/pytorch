#include <ATen/native/MathBitsFallback.h>

namespace at {

struct ConjFallback : MathOpFallback {
  ConjFallback() : MathOpFallback(DispatchKey::Conjugate, "conjugate") {}
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_conj();
  }
  void _set_bit(const Tensor& tensor, bool value) override {
    return tensor._set_conj(value);
  }
  Tensor resolve_bit(const Tensor& tensor) override {
    return at::resolve_conj(tensor);
  }
  Tensor& math_op_(Tensor& tensor) override {
    return at::conj_physical_(tensor);
  }
};

void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  ConjFallback object;
  object.fallback_impl(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("requires_grad_", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj", torch::CppFunction::makeFallthrough());
  m.impl("_conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
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
  m.impl("view_as_real", torch::CppFunction::makeFallthrough());
  m.impl("imag", torch::CppFunction::makeFallthrough());
  m.impl("real", torch::CppFunction::makeFallthrough());
  m.impl("view", torch::CppFunction::makeFallthrough());
  m.impl("_unsafe_view", torch::CppFunction::makeFallthrough());
  m.impl("reshape", torch::CppFunction::makeFallthrough());
}

} // namespace at
