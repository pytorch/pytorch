#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

bool test_device_equality(torch::stable::Device d1, torch::stable::Device d2) {
  return d1 == d2;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_device_equality(Device d1, Device d2) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_equality", TORCH_BOX(&test_device_equality));
}
