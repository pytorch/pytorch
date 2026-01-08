#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

using torch::stable::Device;
using torch::stable::Tensor;

// Test new_empty with all kwargs
Tensor my_new_empty(
    Tensor self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return torch::stable::new_empty(self, size, dtype, layout, device, pin_memory);
}

// Test new_zeros with all kwargs
Tensor my_new_zeros(
    Tensor self,
    torch::headeronly::IntHeaderOnlyArrayRef size,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return torch::stable::new_zeros(self, size, dtype, layout, device, pin_memory);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_new_empty(Tensor self, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
  m.def(
      "my_new_zeros(Tensor self, int[] size, ScalarType? dtype, Layout? layout, Device? device, bool? pin_memory) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_new_empty", TORCH_BOX(&my_new_empty));
  m.impl("my_new_zeros", TORCH_BOX(&my_new_zeros));
}
