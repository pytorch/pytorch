#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

torch::stable::Device test_device_set_index(
    torch::stable::Device device,
    torch::stable::DeviceIndex index) {
  device.set_index(index);
  return device;
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_device_set_index(Device device, DeviceIndex index) -> Device");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_set_index", TORCH_BOX(&test_device_set_index));
}
