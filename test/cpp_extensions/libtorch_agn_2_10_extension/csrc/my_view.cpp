#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

Tensor my_view(Tensor t, torch::headeronly::HeaderOnlyArrayRef<int64_t> size) {
  return view(t, size);
}

Tensor my_view_dtype(Tensor self, torch::headeronly::ScalarType dtype) {
  return view(self, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_view(Tensor t, int[] size) -> Tensor");
  m.def("my_view_dtype(Tensor self, ScalarType dtype) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    STABLE_LIB_NAME,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_view", TORCH_BOX(&my_view));
  m.impl("my_view_dtype", TORCH_BOX(&my_view_dtype));
}
