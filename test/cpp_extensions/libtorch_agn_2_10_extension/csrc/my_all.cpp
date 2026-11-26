#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;
using OptionalDims = std::optional<torch::headeronly::IntHeaderOnlyArrayRef>;

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_all(Tensor self) -> Tensor");
  m.def(
      "my_all_dim(Tensor self, int dim, bool keepdim=False) -> Tensor");
  m.def(
      "my_all_dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_all",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&)>(
          &torch::stable::all)));
  m.impl(
      "my_all_dim",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&, int64_t, bool)>(
          &torch::stable::all)));
  m.impl(
      "my_all_dims",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&, OptionalDims, bool)>(
          &torch::stable::all)));
}
