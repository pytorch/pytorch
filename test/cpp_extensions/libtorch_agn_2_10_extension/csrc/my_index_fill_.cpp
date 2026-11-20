#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

#include <cstdint>

using torch::stable::Tensor;

namespace {

Tensor my_index_fill_(
    Tensor self,
    int64_t dim,
    const Tensor& index,
    const Tensor& value) {
  return torch::stable::index_fill_(self, dim, index, value);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_index_fill_(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_index_fill_", TORCH_BOX(&my_index_fill_));
}
