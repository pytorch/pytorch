#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>

using torch::stable::Tensor;

Tensor my_full(
    std::vector<int64_t> size,
    double fill_value,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout> layout,
    std::optional<torch::stable::Device> device,
    std::optional<bool> pin_memory) {
  return torch::stable::full(size, fill_value, dtype, layout, device, pin_memory);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_full(int[] size, float fill_value, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_full", TORCH_BOX(&my_full));
}
