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

Tensor my_empty_strided(
    torch::headeronly::IntHeaderOnlyArrayRef size,
    torch::headeronly::IntHeaderOnlyArrayRef stride,
    std::optional<torch::headeronly::ScalarType> dtype,
    std::optional<torch::headeronly::Layout> layout,
    std::optional<torch::stable::Device> device,
    std::optional<bool> pin_memory) {
  return torch::stable::empty_strided(
      size, stride, dtype, layout, device, pin_memory);
}

Tensor my_slice(
    const Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end,
    int64_t step) {
  return torch::stable::slice(self, dim, start, end, step);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_empty(int[] size, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
  m.def(
      "my_empty_strided(int[] size, int[] stride, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "my_slice(Tensor self, int dim=0, int? start=None, int? end=None, int step=1) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_empty", TORCH_BOX(&my_empty));
  m.impl("my_empty_strided", TORCH_BOX(&my_empty_strided));
  m.impl("my_slice", TORCH_BOX(&my_slice));
}
