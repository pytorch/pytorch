#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>

using torch::stable::Tensor;

torch::stable::Device test_tensor_device(torch::stable::Tensor tensor) {
  return tensor.device();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_tensor_device(Tensor t) -> Device");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_tensor_device", TORCH_BOX(&test_tensor_device));
}
