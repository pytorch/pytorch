#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

using torch::stable::Tensor;

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def(
      "my_repeat_interleave(Tensor self, int repeats, int? dim=None, *, int? output_size=None) -> Tensor");
  m.def(
      "my_repeat_interleave_tensor(Tensor repeats, *, int? output_size=None) -> Tensor");
  m.def(
      "my_repeat_interleave_self_tensor(Tensor self, Tensor repeats, int? dim=None, *, int? output_size=None) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "my_repeat_interleave",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&, int64_t, std::optional<int64_t>, std::optional<int64_t>)>(
          &torch::stable::repeat_interleave)));
  m.impl(
      "my_repeat_interleave_tensor",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&, std::optional<int64_t>)>(
          &torch::stable::repeat_interleave)));
  m.impl(
      "my_repeat_interleave_self_tensor",
      TORCH_BOX(static_cast<Tensor (*)(const Tensor&, const Tensor&, std::optional<int64_t>, std::optional<int64_t>)>(
          &torch::stable::repeat_interleave)));
}
