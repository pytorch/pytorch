#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_bitwise_and(const Tensor& self, const Tensor& other) {
  return torch::stable::bitwise_and(self, other);
}

Tensor my_bitwise_or(const Tensor& self, const Tensor& other) {
  return torch::stable::bitwise_or(self, other);
}

Tensor my_bitwise_left_shift(const Tensor& self, const Tensor& other) {
  return torch::stable::bitwise_left_shift(self, other);
}

Tensor my_bitwise_right_shift(const Tensor& self, const Tensor& other) {
  return torch::stable::bitwise_right_shift(self, other);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_bitwise_and(Tensor self, Tensor other) -> Tensor");
  m.def("my_bitwise_or(Tensor self, Tensor other) -> Tensor");
  m.def("my_bitwise_left_shift(Tensor self, Tensor other) -> Tensor");
  m.def("my_bitwise_right_shift(Tensor self, Tensor other) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_bitwise_and", TORCH_BOX(&my_bitwise_and));
  m.impl("my_bitwise_or", TORCH_BOX(&my_bitwise_or));
  m.impl("my_bitwise_left_shift", TORCH_BOX(&my_bitwise_left_shift));
  m.impl("my_bitwise_right_shift", TORCH_BOX(&my_bitwise_right_shift));
}
