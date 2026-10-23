#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <optional>
#include <string>

using torch::stable::Tensor;

Tensor my_from_file(
    const std::string& filename,
    std::optional<bool> shared,
    std::optional<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype) {
  return torch::stable::from_file(filename, shared, size, dtype);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_from_file(str filename, bool? shared=None, int? size=0, ScalarType? dtype=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_from_file", TORCH_BOX(&my_from_file));
}
