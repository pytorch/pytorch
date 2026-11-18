#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>

using torch::stable::Tensor;

// torch::stable::subtract using STABLE_TORCH_ERROR_CODE_CHECK (detailed error
// message via the stable ABI) instead of TORCH_ERROR_CODE_CHECK (simple message).
//
// This lives in the 2.10 extension on purpose: built at a target older than 2.13,
// STABLE_TORCH_ERROR_CODE_CHECK reaches torch_exception_get_what* through a
// runtime symbol lookup (TORCH_DYNAMIC_VERSION_CALL), so it exercises the dynamic
// version call path. Built at 2.13+ the same op takes the direct-call shortcut.
inline torch::stable::Tensor our_subtract_stable_error_check(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}

// Same op but with the less detailed TORCH_ERROR_CODE_CHECK, for comparison.
inline torch::stable::Tensor our_subtract_torch_error_check(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("our_subtract_stable_error_check(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
  m.def("our_subtract_torch_error_check(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("our_subtract_stable_error_check", TORCH_BOX(&our_subtract_stable_error_check));
  m.impl("our_subtract_torch_error_check", TORCH_BOX(&our_subtract_torch_error_check));
}
