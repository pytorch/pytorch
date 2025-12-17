#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

#include <string>

torch::stable::Device test_device_constructor(
    bool is_cuda,
    torch::stable::DeviceIndex index,
    bool use_str) {
  using torch::stable::Device;
  using torch::stable::DeviceType;

  if (use_str) {
    std::string device_str;
    if (is_cuda) {
      device_str = "cuda:" + std::to_string(index);
    } else {
      device_str = "cpu";
    }
    return Device(device_str);
  } else {
    if (is_cuda) {
      return Device(DeviceType::CUDA, index);
    } else {
      return Device(DeviceType::CPU);
    }
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "test_device_constructor(bool is_cuda, DeviceIndex index, bool use_str) -> Device");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_constructor", TORCH_BOX(&test_device_constructor));
}
