#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h>

#include <string>

using torch::stable::Tensor;

/// torch::stable::subtract with STABLE_TORCH_ERROR_CODE_CHECK instead of TORCH_ERROR_CODE_CHECK
inline torch::stable::Tensor our_subtract_stable_error_check(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  STABLE_TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}

/// Similar to the function above, but with the less detailed TORCH_ERROR_CODE_CHECK
inline torch::stable::Tensor our_subtract_torch_error_check(
    const torch::stable::Tensor& self,
    const torch::stable::Tensor& other,
    double alpha = 1.0) {
  AtenTensorHandle ret0;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_aten_subtract_Tensor(self.get(), other.get(), alpha, &ret0));
  return torch::stable::Tensor(ret0);
}


std::string my_exception_what() {
  return std::string(torch_exception_get_what());
}
std::string my_exception_get_what_without_backtrace() {
  return std::string(torch_exception_get_what_without_backtrace());
}
bool my_torch_exception_get_exception_printing() {
  return torch_exception_get_exception_printing();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("our_subtract_stable_error_check(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
  m.def("our_subtract_torch_error_check(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
  m.def("my_exception_what() -> str");
  m.def("my_exception_get_what_without_backtrace() -> str");
  m.def("my_torch_exception_get_exception_printing() -> bool");

}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("our_subtract_stable_error_check", TORCH_BOX(&our_subtract_stable_error_check));
  m.impl("our_subtract_torch_error_check", TORCH_BOX(&our_subtract_torch_error_check));
  m.impl("my_exception_what", TORCH_BOX(&my_exception_what));
  m.impl("my_exception_get_what_without_backtrace", TORCH_BOX(&my_exception_get_what_without_backtrace));
  m.impl("my_torch_exception_get_exception_printing", TORCH_BOX(&my_torch_exception_get_exception_printing));
}
