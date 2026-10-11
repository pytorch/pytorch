#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>

bool test_device_is_privateuseone(torch::stable::Device device) {
  return device.is_privateuseone();
}

bool my_is_privateuseone(torch::stable::Tensor t) {
  return t.is_privateuseone();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_device_is_privateuseone(Device device) -> bool");
  m.def("my_is_privateuseone(Tensor t) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "test_device_is_privateuseone",
      TORCH_BOX(&test_device_is_privateuseone));
  m.impl("my_is_privateuseone", TORCH_BOX(&my_is_privateuseone));
}