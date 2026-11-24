#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

bool my_is_conj(Tensor t) {
  return t.is_conj();
}

bool my_is_neg(Tensor t) {
  return t.is_neg();
}

bool my_is_inference(Tensor t) {
  return t.is_inference();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_is_conj(Tensor t) -> bool");
  m.def("my_is_neg(Tensor t) -> bool");
  m.def("my_is_inference(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_is_conj", TORCH_BOX(&my_is_conj));
  m.impl("my_is_neg", TORCH_BOX(&my_is_neg));
  m.impl("my_is_inference", TORCH_BOX(&my_is_inference));
}

// Preserve lazy bits instead of resolving them through the backend fallbacks.
STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, Conjugate, m) {
  m.impl("my_is_conj", TORCH_BOX(&my_is_conj));
  m.impl("my_is_neg", TORCH_BOX(&my_is_neg));
  m.impl("my_is_inference", TORCH_BOX(&my_is_inference));
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, Negative, m) {
  m.impl("my_is_conj", TORCH_BOX(&my_is_conj));
  m.impl("my_is_neg", TORCH_BOX(&my_is_neg));
  m.impl("my_is_inference", TORCH_BOX(&my_is_inference));
}
