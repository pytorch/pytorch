#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

Tensor my_sum(
    Tensor self,
    torch::headeronly::HeaderOnlyArrayRef<int64_t> dim,
    bool keepdim = false,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  // Check if dim is empty to determine if we should sum all dimensions
  if (dim.size() == 0) {
    return sum(self, std::nullopt, keepdim, dtype);
  }
  return sum(self, dim, keepdim, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def("my_sum(Tensor self, int[] dim, bool keepdim=False, ScalarType? dtype=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agnostic_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_sum", TORCH_BOX(&my_sum));
}
