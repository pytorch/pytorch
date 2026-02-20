#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>

// This is used to test const torch::stable::Device& with TORCH_BOX
bool test_device_is_cuda(const torch::stable::Device& device) {
  return device.is_cuda();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_device_is_cuda(Device device) -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_device_is_cuda", TORCH_BOX(&test_device_is_cuda));
}
