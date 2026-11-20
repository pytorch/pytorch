#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;
using MedianResult = std::tuple<Tensor, Tensor>;

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_median(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)");
  m.def("my_median_default(Tensor self) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_median",
      TORCH_BOX(static_cast<MedianResult (*)(const Tensor&, int64_t, bool)>(
          &torch::stable::median)));
  m.impl(
      "my_median_default",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&)>(
          &torch::stable::median)));
}
