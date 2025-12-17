#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <vector>

using torch::stable::Tensor;

// Declare my__foreach_mul (defined in my__foreach_mul.cpp)
extern std::vector<Tensor> my__foreach_mul(
    const torch::headeronly::HeaderOnlyArrayRef<Tensor>& self,
    torch::headeronly::HeaderOnlyArrayRef<Tensor> other);

// Helper function for cloning
Tensor my_clone(Tensor t) {
  return clone(t);
}

std::vector<Tensor> make_tensor_clones_and_call_foreach(Tensor t1, Tensor t2) {
  // This function tests that my__foreach_mul can take in std::initializer_lists
  // in addition to std::vectors.
  Tensor t1_1 = my_clone(t1);
  Tensor t1_2 = my_clone(t1);
  Tensor t2_1 = my_clone(t2);
  Tensor t2_2 = my_clone(t2);
  return my__foreach_mul({t1_1, t2_1}, {t1_2, t2_2});
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def(
      "make_tensor_clones_and_call_foreach(Tensor t1, Tensor t2) -> Tensor[]");
}

STABLE_TORCH_LIBRARY_IMPL(
    libtorch_agn_2_10,
    CompositeExplicitAutograd,
    m) {
  m.impl(
      "make_tensor_clones_and_call_foreach",
      TORCH_BOX(&make_tensor_clones_and_call_foreach));
}
