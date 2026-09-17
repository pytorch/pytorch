#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor my_mm_out(Tensor out, Tensor self, Tensor mat2) {
  return torch::stable::mm_out(out, self, mat2);
}

Tensor my_bmm_out(Tensor out, Tensor self, Tensor mat2) {
  return torch::stable::bmm_out(out, self, mat2);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_mm_out(Tensor(a!) out, Tensor self, Tensor mat2) -> Tensor(a!)");
  m.def("my_bmm_out(Tensor(a!) out, Tensor self, Tensor mat2) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_mm_out", TORCH_BOX(&my_mm_out));
  m.impl("my_bmm_out", TORCH_BOX(&my_bmm_out));
}
