#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>

#include <optional>

using torch::stable::Tensor;

Tensor my_empty(
    torch::headeronly::HeaderOnlyArrayRef<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::stable::Device> device,
    std::optional<bool> pin_memory) {
  return empty(size, dtype, device, pin_memory);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agnostic_2_10, m) {
  m.def(
      "my_empty(int[] size, ScalarType? dtype=None, Device? device=None, bool? pin_memory=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agnostic_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_empty", TORCH_BOX(&my_empty));
}
