#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

torch::stable::DeviceIndex test_device_index(torch::stable::Device device) {
  return device.index();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_device_index(Device device) -> DeviceIndex");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_index", TORCH_BOX(&test_device_index));
}
