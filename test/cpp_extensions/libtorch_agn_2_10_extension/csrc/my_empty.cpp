#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>

#include <optional>

using torch::stable::Tensor;

Tensor my_empty(
    torch::headeronly::HeaderOnlyArrayRef<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout>& layout,
    const std::optional<torch::stable::Device>& device,
    std::optional<bool> pin_memory,
    std::optional<torch::headeronly::MemoryFormat> memory_format) {
  return empty(size, dtype, layout, device, pin_memory, memory_format);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "my_empty(int[] size, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my_empty", TORCH_BOX(&my_empty));
}
