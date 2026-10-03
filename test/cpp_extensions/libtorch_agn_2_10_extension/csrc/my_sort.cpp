#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <tuple>

using torch::stable::Tensor;

std::tuple<Tensor, Tensor> my_sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  return torch::stable::sort(self, dim, descending);
}

std::tuple<Tensor, Tensor> my_sort_stable(
    const Tensor& self,
    std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  return torch::stable::sort(self, stable, dim, descending);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_sort(Tensor self, int dim, bool descending) -> (Tensor values, "
      "Tensor indices)");
  m.def(
      "my_sort_stable(Tensor self, bool? stable, int dim, bool descending) "
      "-> (Tensor values, Tensor indices)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_sort", TORCH_BOX(&my_sort));
  m.impl("my_sort_stable", TORCH_BOX(&my_sort_stable));
}
