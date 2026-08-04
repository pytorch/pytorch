#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_view_dtype(Tensor self, torch::headeronly::ScalarType dtype) {
  return torch::stable::view_dtype(self, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_view_dtype(Tensor self, ScalarType dtype) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_view_dtype", TORCH_BOX(&my_view_dtype));
}
