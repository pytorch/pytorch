#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_flip(
    const Tensor& self,
    torch::headeronly::IntHeaderOnlyArrayRef dims) {
  return torch::stable::flip(self, dims);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_flip(Tensor self, int[] dims) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_flip", TORCH_BOX(&my_flip));
}
