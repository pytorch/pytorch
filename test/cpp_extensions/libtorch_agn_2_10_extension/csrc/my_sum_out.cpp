#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

Tensor my_sum_out(
    Tensor out,
    Tensor self,
    std::optional<torch::headeronly::HeaderOnlyArrayRef<int64_t>> dim,
    bool keepdim = false,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  return sum_out(out, self, dim, keepdim, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_sum_out(Tensor(a!) out, Tensor self, int[]? dim=None, bool keepdim=False, ScalarType? dtype=None) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_sum_out", TORCH_BOX(&my_sum_out));
}
