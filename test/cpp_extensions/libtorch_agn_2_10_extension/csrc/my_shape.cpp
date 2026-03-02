#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

torch::headeronly::HeaderOnlyArrayRef<int64_t> my_shape(Tensor t) {
  return t.sizes();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_shape(Tensor t) -> int[]");
}

STABLE_TORCH_LIBRARY_IMPL(
    STABLE_LIB_NAME,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_shape", TORCH_BOX(&my_shape));
}
