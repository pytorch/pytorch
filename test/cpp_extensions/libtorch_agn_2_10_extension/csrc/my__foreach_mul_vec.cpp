#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <vector>

using torch::stable::Tensor;

// This is used to test const std::vector<T>& with TORCH_BOX
std::vector<Tensor> my__foreach_mul_vec(
    const std::vector<Tensor>& self,
    const std::vector<Tensor>& other) {
  std::array<StableIValue, 2> stack = {
      torch::stable::detail::from(self), torch::stable::detail::from(other)};
  aoti_torch_call_dispatcher("aten::_foreach_mul", "List", stack.data());
  return torch::stable::detail::to<std::vector<Tensor>>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("my__foreach_mul_vec(Tensor[] self, Tensor[] other) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl("my__foreach_mul_vec", TORCH_BOX(&my__foreach_mul_vec));
}
