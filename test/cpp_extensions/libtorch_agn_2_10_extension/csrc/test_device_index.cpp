#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

torch::stable::DeviceIndex test_device_index(torch::stable::Device device) {
  return device.index();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_device_index(Device device) -> DeviceIndex");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("test_device_index", TORCH_BOX(&test_device_index));
}
