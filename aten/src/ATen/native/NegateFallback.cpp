#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
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
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFallthrough());

  // See test_metadata_check_when_primal_has_neg_bit in test_autograd.py
  m.impl("_has_same_storage_numel", torch::CppFunction::makeFallthrough());
  m.impl("_new_zeros_with_same_feature_meta", torch::CppFunction::makeFallthrough());

  // linear algebra functions
  m.impl("linalg_solve_triangular", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd.U", torch::CppFunction::makeFallthrough());

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)
}

}
} // namespace at
