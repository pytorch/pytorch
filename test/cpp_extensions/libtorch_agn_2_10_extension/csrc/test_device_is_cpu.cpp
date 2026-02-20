#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

// This is used to test torch::stable::Device& with TORCH_BOX
bool test_device_is_cpu(torch::stable::Device& device) {
  return device.is_cpu();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_device_is_cpu(Device device) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_is_cpu", TORCH_BOX(&test_device_is_cpu));
}
