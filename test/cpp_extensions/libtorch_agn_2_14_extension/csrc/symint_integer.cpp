#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <array>
#include <vector>

namespace {

using torch::stable::detail::from;
using torch::stable::detail::to;

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
using IntegerSymInt = torch::stable::SymInt;
#else
using IntegerSymInt = int64_t;
#endif

IntegerSymInt integer_symint_echo(IntegerSymInt value) {
  return value;
}

int64_t integer_symint_dispatch(int64_t value) {
  std::array<StableIValue, 1> stack{from(value)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      C10_STRINGIZE(STABLE_LIB_NAME) "::integer_symint_echo",
      "",
      stack.data(),
      TORCH_ABI_VERSION));
  return to<int64_t>(stack[0]);
}

torch::stable::Tensor integer_symint_select(
    torch::stable::Tensor self,
    int64_t index) {
  return torch::stable::select(self, 0, index);
}

torch::stable::Tensor integer_symint_view(
    torch::stable::Tensor self,
    const std::vector<int64_t>& sizes) {
  return torch::stable::view(self, sizes);
}

} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("integer_symint_echo(SymInt value) -> SymInt");
  m.def("integer_symint_dispatch(int value) -> int");
  m.def("integer_symint_select(Tensor self, int index) -> Tensor");
  m.def("integer_symint_view(Tensor self, int[] sizes) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeImplicitAutograd, m) {
  m.impl("integer_symint_echo", TORCH_BOX(&integer_symint_echo));
  m.impl("integer_symint_dispatch", TORCH_BOX(&integer_symint_dispatch));
  m.impl("integer_symint_select", TORCH_BOX(&integer_symint_select));
  m.impl("integer_symint_view", TORCH_BOX(&integer_symint_view));
}
