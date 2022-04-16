#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseBlasImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/scalar_tensor_native.h>
#endif

#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {

Tensor& addmv_out_sparse_csr(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.is_sparse_csr());

  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta.toComplexDouble();

  if (&result != &self) {
    at::native::resize_output(result, self_->sizes());
    if (betaval != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  if (mat._nnz() == 0) {
    // shortcut for an empty matrix
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (betaval == 0.0) {
      return result.zero_();
    } else {
      return at::mul_out(
          const_cast<Tensor&>(result),
          self,
          at::native::scalar_tensor(
              beta,
              self.scalar_type(),
              c10::nullopt /*layout*/,
              at::kCPU,
              c10::nullopt /* pin_memory */));
    }
  }

  sparse::impl::cpu::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
  return result;
}

/*
  Solves a system of linear equations whose coefficients are represented in a sparse triangular matrix A:
  op(A) X = B.

  Args:
  * `B` - dense Tensor of size m × nrhs.
  * `A` - sparse Tensor of size m × m.
  * `upper` - controls whether upper or lower triangular part of A is considered in computations.
  * `transpose` - if true then op(A) = A^T.
  * `unitriangular` - if true then the diagonal elements of A are assumed to be one.
  * `X` - dense Tensor of size m × nrhs.
  * `clone_A` - cloned matrix A, required only for compatibility with strided layout interface.
*/
std::tuple<Tensor&, Tensor&> triangular_solve_out_sparse_csr_cpu(
    const Tensor& B,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    Tensor& X,
    Tensor& clone_A) {
  sparse::impl::cpu::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
  return std::tuple<Tensor&, Tensor&>(X, clone_A);
}

} // namespace native
} // namespace at
