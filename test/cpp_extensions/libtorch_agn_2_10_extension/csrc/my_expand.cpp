#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_expand", TORCH_BOX(&torch::stable::expand));
}
