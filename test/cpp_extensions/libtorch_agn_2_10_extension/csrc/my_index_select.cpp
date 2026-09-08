#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_index_select(const Tensor& self, int64_t dim, const Tensor& index) {
  return torch::stable::index_select(self, dim, index);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_index_select(Tensor self, int dim, Tensor index) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_index_select", TORCH_BOX(&my_index_select));
}
