#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_sum(
    Tensor self,
    std::optional<torch::headeronly::HeaderOnlyArrayRef<int64_t>> dim,
    bool keepdim,
    std::optional<torch::headeronly::ScalarType> dtype) {
  return torch::stable::sum(self, dim, keepdim, dtype);
}

// Tests that sum(t) works (independent from the STABLE_TORCH_LIBRARY
// registration which passes a default)
Tensor my_sum_all(Tensor self) {
  return torch::stable::sum(self);
}

// Test op that takes only a tensor and passes [1] as dim
// (sums along dimension 1)
Tensor my_sum_dim1(Tensor self) {
  return torch::stable::sum(
      self,
      std::make_optional(torch::headeronly::IntHeaderOnlyArrayRef({1})),
      false,
      std::nullopt);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_sum(Tensor self, int[]? dim=None, bool keepdim=False, ScalarType? dtype=None) -> Tensor");
  m.def("my_sum_all(Tensor self) -> Tensor");
  m.def("my_sum_dim1(Tensor self) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_sum", TORCH_BOX(&my_sum));
  m.impl("my_sum_all", TORCH_BOX(&my_sum_all));
  m.impl("my_sum_dim1", TORCH_BOX(&my_sum_dim1));
}
