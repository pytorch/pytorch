#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

Tensor identity_with_fake_module(Tensor t) {
  return t;
}

Tensor identity_w_nonexistent_fake_module(Tensor t) {
  return t;
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.set_python_module("libtorch_agn_2_13");
  m.def("identity_with_fake_module(Tensor t) -> Tensor");
}

// Separate fragment whose set_python_module points at a module that doesn't
// exist, to exercise the missing-module error path.
STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.set_python_module("libtorch_agn_2_13_nonexistent_module");
  m.def("identity_w_nonexistent_fake_module(Tensor t) -> Tensor");
}

// Register on CPU key so there's no Meta kernel to fall back to.
STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CPU, m) {
  m.impl("identity_with_fake_module", TORCH_BOX(&identity_with_fake_module));
  m.impl(
      "identity_w_nonexistent_fake_module",
      TORCH_BOX(&identity_w_nonexistent_fake_module));
}
