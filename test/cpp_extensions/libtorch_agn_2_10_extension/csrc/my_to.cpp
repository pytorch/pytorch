#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Device;
using torch::stable::Tensor;

// Test to(device) convenience overload
Tensor my_to_device(Tensor self, Device device) {
  return torch::stable::to(self, device);
}

// Test to(dtype)
Tensor my_to_dtype(Tensor self, torch::headeronly::ScalarType dtype) {
  return torch::stable::to(self, dtype);
}

// Test the full to.dtype_layout op with all parameters
Tensor my_to_dtype_layout(
    Tensor self,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    bool copy,
    std::optional<torch::headeronly::MemoryFormat> memory_format) {
  return torch::stable::to(
      self,
      dtype,
      layout,
      device,
      pin_memory,
      non_blocking,
      copy,
      memory_format);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my_to_device(Tensor self, Device device) -> Tensor");
  m.def("my_to_dtype(Tensor self, ScalarType dtype) -> Tensor");
  m.def(
      "my_to_dtype_layout(Tensor self, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory, bool non_blocking, bool copy, MemoryFormat? memory_format) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my_to_device", TORCH_BOX(&my_to_device));
  m.impl("my_to_dtype", TORCH_BOX(&my_to_dtype));
  m.impl("my_to_dtype_layout", TORCH_BOX(&my_to_dtype_layout));
}
