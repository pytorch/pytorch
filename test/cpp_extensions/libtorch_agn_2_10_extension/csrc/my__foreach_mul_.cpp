#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

using torch::stable::Tensor;

void my__foreach_mul_(torch::headeronly::HeaderOnlyArrayRef<Tensor> self, torch::headeronly::HeaderOnlyArrayRef<Tensor> other) {
  std::array<StableIValue, 2> stack = {torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul_", "List", stack.data());
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my__foreach_mul_(Tensor(a!)[] self, Tensor[] other) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("my__foreach_mul_", TORCH_BOX(&my__foreach_mul_));
}
