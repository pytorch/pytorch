#include <ATen/native/MathBitFallThroughLists.h>
#include <ATen/view/UnaryInvolutionFallback.h>

namespace at::native {

TORCH_LIBRARY_IMPL(_, Conjugate, m) {
  at::view::register_unary_involution_fallback<c10::DispatchKey::Conjugate>(m);
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
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFallthrough());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFallthrough());

  // See test_metadata_check_when_primal_has_conj_bit in test_autograd.py
  m.impl("_has_same_storage_numel", torch::CppFunction::makeFallthrough());
  m.impl("_new_zeros_with_same_feature_meta", torch::CppFunction::makeFallthrough());

  // linear algebra functions
  m.impl("dot", torch::CppFunction::makeFallthrough());
  m.impl("vdot", torch::CppFunction::makeFallthrough());
  m.impl("dot.out", torch::CppFunction::makeFallthrough());
  m.impl("vdot.out", torch::CppFunction::makeFallthrough());
  m.impl("mm", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular", torch::CppFunction::makeFallthrough());
  m.impl("linalg_solve_triangular.out", torch::CppFunction::makeFallthrough());
  m.impl("mm.out", torch::CppFunction::makeFallthrough());
  m.impl("addmm", torch::CppFunction::makeFallthrough());
  m.impl("addmm_", torch::CppFunction::makeFallthrough());
  m.impl("addmm.out", torch::CppFunction::makeFallthrough());
  m.impl("bmm", torch::CppFunction::makeFallthrough());
  m.impl("bmm.out", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm_", torch::CppFunction::makeFallthrough());
  m.impl("baddbmm.out", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd", torch::CppFunction::makeFallthrough());
  m.impl("linalg_svd.U", torch::CppFunction::makeFallthrough());

  TORCH_VIEW_FNS(m)
  TENSOR_UTILITIES_AND_CONSTRUCTORS(m)
  TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)
}

} // namespace at::native
