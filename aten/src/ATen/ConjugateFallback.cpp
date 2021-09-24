#include <ATen/native/MathBitsFallback.h>
#include <ATen/native/MathBitFallThroughLists.h>

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

void conjugateFallbackToHandleOnlyMutableInputs(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  ConjFallback object;
  object.linalg_fallback(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&conjugateFallback>());
}

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("set_.source_Storage_storage_offset", torch::CppFunction::makeFallthrough());
  m.impl("set_.source_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("set_", torch::CppFunction::makeFallthrough());
  m.impl("copy_", torch::CppFunction::makeFallthrough());
  m.impl("clone", torch::CppFunction::makeFallthrough());
  m.impl("_conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical", torch::CppFunction::makeFallthrough());
  m.impl("conj_physical_", torch::CppFunction::makeFallthrough());
  m.impl("resolve_conj", torch::CppFunction::makeFallthrough());
  m.impl("resolve_neg", torch::CppFunction::makeFallthrough());

  // linear algebra functions
  m.impl("dot", torch::CppFunction::makeFallthrough());
  m.impl("vdot", torch::CppFunction::makeFallthrough());
  m.impl("dot.out", torch::CppFunction::makeFallthrough());
  m.impl("vdot.out", torch::CppFunction::makeFallthrough());
  m.impl("mm", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", torch::CppFunction::makeFromBoxedFunction<&conjugateFallbackToHandleOnlyMutableInputs>());
  m.impl("mm.out", torch::CppFunction::makeFallthrough());
  m.impl("addmm", torch::CppFunction::makeFallthrough());
  m.impl("addmm_", torch::CppFunction::makeFallthrough());
  m.impl("addmm.out", torch::CppFunction::makeFallthrough());
  m.impl("bmm", torch::CppFunction::makeFallthrough());
  m.impl("bmm.out", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm_", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm.out", torch::CppFunction::makeFallthrough());

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
}

} // namespace at
