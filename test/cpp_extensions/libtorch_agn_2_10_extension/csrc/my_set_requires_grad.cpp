#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_set_requires_grad(Tensor t, bool requires_grad) {
  t.set_requires_grad(requires_grad);
  return t;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_set_requires_grad(Tensor t, bool requires_grad) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_set_requires_grad", TORCH_BOX(&my_set_requires_grad));
}
