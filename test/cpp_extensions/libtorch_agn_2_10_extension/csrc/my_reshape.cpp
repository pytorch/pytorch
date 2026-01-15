#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

Tensor my_reshape(Tensor t, torch::headeronly::HeaderOnlyArrayRef<int64_t> shape) {
  return reshape(t, shape);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_reshape(Tensor t, int[] shape) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_reshape", TORCH_BOX(&my_reshape));
}
