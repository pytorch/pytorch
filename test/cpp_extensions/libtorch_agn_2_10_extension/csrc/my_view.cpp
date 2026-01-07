#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

Tensor my_view(Tensor t, torch::headeronly::HeaderOnlyArrayRef<int64_t> size) {
  return view(t, size);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_view(Tensor t, int[] size) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_view", TORCH_BOX(&my_view));
}
