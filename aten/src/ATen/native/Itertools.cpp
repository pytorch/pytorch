#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/cartesian_prod_native.h>
#include <ATen/ops/combinations_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/index.h>
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/nonzero_static.h>
#include <ATen/ops/stack.h>
#endif

#include <vector>

namespace {

using namespace at;

c10::SymInt calculate_num_combinations(c10::SymInt n, int64_t r, bool with_replacement) {
  if (r < 0) return 0;
  if (r == 0) return 1;
  if (with_replacement) {
    n = n + r - 1;
  }
  c10::SymInt res = 1;
  for (int64_t i = 1; i <= r; ++i) {
    res = res * (n - r + i) / i;
  }
  return res;
}

Tensor _triu_mask(c10::SymInt n, int64_t dims, bool diagonal, TensorOptions opt) {
  // get a mask that has value 1 whose indices satisfies i < j < k < ...
  // or i <= j <= k <= ... (depending on diagonal)
  Tensor range = at::arange(std::move(n), opt.dtype(kLong));
  std::vector<Tensor> index_grids = at::meshgrid(std::vector<Tensor>(dims, range), "ij");
  Tensor mask = at::full_symint(index_grids[0].sym_sizes(), true, opt.dtype(kBool));
  if(diagonal) {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] <= index_grids[i+1];
    }
  } else {
    for(int64_t i = 0; i < dims - 1; i++) {
      mask *= index_grids[i] < index_grids[i+1];
    }
  }
  return mask;
}

}  // namespace

namespace at::native {

Tensor cartesian_prod(TensorList tensors) {
  for(const Tensor &t : tensors) {
    TORCH_CHECK(t.dim() == 1, "Expect a 1D vector, but got shape ", t.sizes());
  }
  if (tensors.size() == 1) {
    return tensors[0];
  }
  std::vector<Tensor> grids = at::meshgrid(tensors, "ij");
  for(Tensor &t : grids) {
    t = t.flatten();
  }
  return at::stack(grids, 1);
}

Tensor combinations(const Tensor& self, int64_t r, bool with_replacement) {
  TORCH_CHECK(self.dim() == 1, "Expect a 1D vector, but got shape ", self.sizes());
  TORCH_CHECK(r >= 0, "Expect a non-negative number, but got ", r);
  if (r == 0) {
    return at::empty({0}, self.options());
  }
  c10::SymInt num_elements = self.sym_numel();
  c10::SymInt num_combinations = calculate_num_combinations(num_elements, r, with_replacement);
  Tensor indices;
  {
    Tensor mask = _triu_mask(std::move(num_elements), r, with_replacement, self.options());
    indices = at::nonzero_static_symint(mask, num_combinations);
  }
  return self.index({indices}).contiguous();
}

}  // namespace at::native
