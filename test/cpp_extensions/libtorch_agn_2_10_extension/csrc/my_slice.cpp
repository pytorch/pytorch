#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include <cstdint>
#include <optional>

using torch::stable::Tensor;

Tensor my_slice(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  return torch::stable::slice(self, dim, start, end, step);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_slice(Tensor(a) self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor(a)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_slice", TORCH_BOX(&my_slice));
}
