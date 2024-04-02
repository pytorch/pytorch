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
#include <ATen/ops/meshgrid.h>
#include <ATen/ops/stack.h>
#endif

#include <vector>

namespace {

using namespace at;

Tensor _triu_mask(int64_t n, int64_t dims, bool diagonal, TensorOptions opt) {
  // get a mask that has value 1 whose indices satisfies i < j < k < ...
  // or i <= j <= k <= ... (depending on diagonal)
  Tensor range = at::arange(n, opt.dtype(kLong));
  std::vector<Tensor> index_grids = at::meshgrid(std::vector<Tensor>(dims, range), "ij");
  Tensor mask = at::full(index_grids[0].sizes(), true, opt.dtype(kBool));
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

namespace at {
namespace native{

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
  int64_t num_elements = self.numel();
  std::vector<Tensor> grids = at::meshgrid(std::vector<Tensor>(r, self), "ij");
  Tensor mask = _triu_mask(num_elements, r, with_replacement, self.options());
  for(Tensor &t : grids) {
    t = t.masked_select(mask);
  }
  return at::stack(grids, 1);
}

}  // namespace native
}  // namespace at
