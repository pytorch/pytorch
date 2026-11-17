#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/symint.h>
#include <torch/csrc/stable/tensor.h>

#include <array>
#include <optional>
#include <vector>

namespace {

using torch::stable::SymInt;
using torch::stable::Tensor;
using torch::stable::detail::from;
using torch::stable::detail::to;

Tensor symint_view(Tensor self, const std::vector<SymInt>& sizes) {
  std::array<StableIValue, 2> stack{from(self), from(sizes)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::view", "", stack.data(), TORCH_ABI_VERSION));
  return to<Tensor>(stack[0]);
}

Tensor symint_slice(
    Tensor self,
    int64_t dim,
    std::optional<SymInt> start,
    std::optional<SymInt> end,
    SymInt step) {
  std::array<StableIValue, 5> stack{
      from(self), from(dim), from(start), from(end), from(step)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::slice", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return to<Tensor>(stack[0]);
}

SymInt symint_size(Tensor self, int64_t dim) {
  std::array<StableIValue, 2> stack{from(self), from(dim)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::sym_size", "int", stack.data(), TORCH_ABI_VERSION));
  return to<SymInt>(stack[0]);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("symint_view(Tensor(a) self, SymInt[] sizes) -> Tensor(a)");
  m.def(
      "symint_slice(Tensor(a) self, int dim, SymInt? start, SymInt? end, SymInt step) -> Tensor(a)");
  m.def("symint_size(Tensor self, int dim) -> SymInt");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeImplicitAutograd, m) {
  m.impl("symint_view", TORCH_BOX(&symint_view));
  m.impl("symint_slice", TORCH_BOX(&symint_slice));
  m.impl("symint_size", TORCH_BOX(&symint_size));
}
