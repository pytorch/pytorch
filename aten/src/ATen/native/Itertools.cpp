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

// Helper function to create upper triangular mask for combinations.
// Uses c10::SymInt to support dynamic shapes during tracing/compilation.
// Fixes https://github.com/pytorch/pytorch/issues/163759
Tensor _triu_mask(c10::SymInt n, int64_t dims, bool diagonal, TensorOptions opt) {
  // get a mask that has value 1 whose indices satisfies i < j < k < ...
  // or i <= j <= k <= ... (depending on diagonal)
  Tensor range = at::arange(n, opt.dtype(kLong));
  std::vector<Tensor> index_grids = at::meshgrid(std::vector<Tensor>(dims, range), "ij");
  // Use sym_sizes() instead of sizes() to support symbolic shapes
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

// Compute binomial coefficient C(n, r) using SymInt to support symbolic shapes
// Uses iterative formula: C(n,r) = n*(n-1)*...*(n-r+1) / (r*(r-1)*...*1)
// For combinations with replacement, computes C(n+r-1, r)
c10::SymInt compute_binomial_coeff(c10::SymInt n, int64_t r, bool with_replacement) {
  if (with_replacement) {
    // C(n+r-1, r) for combinations with replacement
    n = n + (r - 1);
  }

  if (r == 0) {
    return c10::SymInt(1);
  }

  c10::SymInt result(1);
  for (int64_t i = 0; i < r; ++i) {
    // This division is always exact because the product of (i+1)
    // consecutive integers is always divisible by (i+1)!
    result = result * (n - i) / (i + 1);
  }
  return result;
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
  // Use sym_size(0) instead of numel() to support dynamic shapes during
  // tracing/compilation. Since we already checked dim() == 1, sym_size(0)
  // gives us the same result as sym_numel() but is more explicit.
  // Fixes https://github.com/pytorch/pytorch/issues/163759
  c10::SymInt num_elements = self.sym_size(0);

  // Compute expected output size using binomial coefficient
  // This helps symbolic shape inference understand the output shape
  c10::SymInt expected_combinations = compute_binomial_coeff(num_elements, r, with_replacement);

  std::vector<Tensor> grids = at::meshgrid(std::vector<Tensor>(r, self), "ij");
  Tensor mask = _triu_mask(num_elements, r, with_replacement, self.options());
  for(Tensor &t : grids) {
    t = t.masked_select(mask);
    // Assert that masked_select returned the expected size
    // This helps symbolic shape inference and catches bugs
    TORCH_INTERNAL_ASSERT(t.sym_size(0) == expected_combinations,
                          "combinations: unexpected output size from masked_select, "
                          "got ", t.sym_size(0), " but expected ", expected_combinations);
  }
  return at::stack(grids, 1);
}

}  // namespace at::native
