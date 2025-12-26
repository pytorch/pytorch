#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_subtract(const Tensor& self, const Tensor& other, double alpha) {
  return torch::stable::subtract(self, other, alpha);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_subtract(Tensor self, Tensor other, float alpha=1.0) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_subtract", TORCH_BOX(&my_subtract));
}
