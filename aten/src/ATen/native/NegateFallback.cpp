#include <ATen/native/MathBitsFallback.h>
#include <ATen/native/MathBitFallThroughLists.h>

namespace at {
namespace native {
struct NegFallback : MathOpFallback {
  NegFallback() : MathOpFallback(DispatchKey::Negative, "negation") {}
  bool is_bit_set(const Tensor& tensor) override {
    return tensor.is_neg();
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
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("neg_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_neg", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
}

}
} // namespace at
