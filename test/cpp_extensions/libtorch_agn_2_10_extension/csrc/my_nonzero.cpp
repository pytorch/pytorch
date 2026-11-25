#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_nonzero(Tensor self) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_nonzero", TORCH_BOX(&torch::stable::nonzero));
}
