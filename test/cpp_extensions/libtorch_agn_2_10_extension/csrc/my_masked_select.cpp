#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_masked_select(const Tensor& self, const Tensor& mask) {
  return torch::stable::masked_select(self, mask);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_masked_select(Tensor self, Tensor mask) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_masked_select", TORCH_BOX(&my_masked_select));
}
