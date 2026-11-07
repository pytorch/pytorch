#include <array>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/scalar.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Scalar;
using torch::stable::Tensor;

Scalar my_scalar_identity(Scalar value) {
  return value;
}

std::vector<Scalar> my_scalar_list(std::vector<Scalar> values) {
  return values;
}

std::optional<Scalar> my_scalar_optional(std::optional<Scalar> value) {
  return value;
}

std::tuple<bool, bool, bool, bool> my_scalar_kind(Scalar value) {
  return {
      value.isBoolean(),
      value.isIntegral(false),
      value.isFloatingPoint(),
      value.isComplex()};
}

Scalar my_scalar_from_int(int64_t value) {
  return Scalar(value);
}

Scalar my_scalar_from_double(double value) {
  return Scalar(value);
}

Scalar my_scalar_from_bool(bool value) {
  return Scalar(value);
}

Scalar my_scalar_from_complex(double real, double imag) {
  return Scalar(c10::complex<double>(real, imag));
}

std::tuple<int64_t, double, bool> my_scalar_to_real(Scalar value) {
  return {value.toLong(), value.toDouble(), value.toBool()};
}

std::tuple<double, double> my_scalar_to_complex(Scalar value) {
  auto result = value.toComplexDouble();
  return {result.real(), result.imag()};
}

Tensor my_scalar_add(Tensor self, Tensor other, Scalar alpha) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(other),
      torch::stable::detail::from(alpha)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::add", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

Scalar my_scalar_item(Tensor self) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::item", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Scalar>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_scalar_identity(Scalar value) -> Scalar");
  m.def("my_scalar_list(Scalar[] values) -> Scalar[]");
  m.def("my_scalar_optional(Scalar? value) -> Scalar?");
  m.def("my_scalar_kind(Scalar value) -> (bool, bool, bool, bool)");
  m.def("my_scalar_from_int(int value) -> Scalar");
  m.def("my_scalar_from_double(float value) -> Scalar");
  m.def("my_scalar_from_bool(bool value) -> Scalar");
  m.def("my_scalar_from_complex(float real, float imag) -> Scalar");
  m.def("my_scalar_to_real(Scalar value) -> (int, float, bool)");
  m.def("my_scalar_to_complex(Scalar value) -> (float, float)");
  m.def("my_scalar_add(Tensor self, Tensor other, Scalar alpha) -> Tensor");
  m.def("my_scalar_item(Tensor self) -> Scalar");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_scalar_identity", TORCH_BOX(&my_scalar_identity));
  m.impl("my_scalar_list", TORCH_BOX(&my_scalar_list));
  m.impl("my_scalar_optional", TORCH_BOX(&my_scalar_optional));
  m.impl("my_scalar_kind", TORCH_BOX(&my_scalar_kind));
  m.impl("my_scalar_from_int", TORCH_BOX(&my_scalar_from_int));
  m.impl("my_scalar_from_double", TORCH_BOX(&my_scalar_from_double));
  m.impl("my_scalar_from_bool", TORCH_BOX(&my_scalar_from_bool));
  m.impl("my_scalar_from_complex", TORCH_BOX(&my_scalar_from_complex));
  m.impl("my_scalar_to_real", TORCH_BOX(&my_scalar_to_real));
  m.impl("my_scalar_to_complex", TORCH_BOX(&my_scalar_to_complex));
  m.impl("my_scalar_add", TORCH_BOX(&my_scalar_add));
  m.impl("my_scalar_item", TORCH_BOX(&my_scalar_item));
}
