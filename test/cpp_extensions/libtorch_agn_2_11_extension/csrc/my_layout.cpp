#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::headeronly::Layout;
using torch::stable::Tensor;

bool my_layout(const Tensor& t, Layout layout) {
  return t.layout() == layout;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_layout(Tensor t, Layout layout) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(
    STABLE_LIB_NAME,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_layout", TORCH_BOX(&my_layout));
}
