#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {

/*
  Computes `result` <- α*(A @ B) * spy(C) + β*C, where spy(C) is the sparsity pattern matrix of C.

  Args:
  * `mat1` - [in] dense Tensor A of size m × k.
  * `mat2` - [in] dense Tensor B of size k × n.
  * `self` - [in] sparse Tensor C of size m × n.
  * `result` - [out] sparse Tensor of size m × n.
*/
Tensor& sparse_sampled_addmm_out_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.is_sparse_csr());

  TORCH_CHECK(mat1.layout() == kStrided, "sampled_addmm: Expected mat1 to have strided layout, but got ", mat1.layout());
  TORCH_CHECK(mat2.layout() == kStrided, "sampled_addmm: Expected mat2 to have strided layout, but got ", mat2.layout());

  TORCH_CHECK(result.layout() == kSparseCsr, "sampled_addmm: Expected result to have sparse csr layout, but got ", result.layout());

  TORCH_CHECK(mat1.scalar_type() == mat2.scalar_type(), "sampled_addmm: Expected mat1 and mat2 to have the same dtype, but got ", mat1.scalar_type(), " and ", mat2.scalar_type());
  TORCH_CHECK(mat1.scalar_type() == self.scalar_type(), "sampled_addmm: Expected mat1 and self to have the same dtype, but got ", mat1.scalar_type(), " and ", self.scalar_type());
  TORCH_CHECK(result.scalar_type() == self.scalar_type(), "sampled_addmm: Expected result and self to have the same dtype, but got ", result.scalar_type(), " and ", self.scalar_type());

  TORCH_CHECK(
      mat1.dim() == 2, "sampled_addmm: Expected mat1 to be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "sampled_addmm: Expected mat2 to be a matrix, got ", mat2.dim(), "-D tensor");
  TORCH_CHECK(
    result.dim() == 2, "sampled_addmm: Expected result to be a matrix, got ", result.dim(), "-D tensor");

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(
      mat1_sizes[1] == mat2_sizes[0],
      "sampled_addmm: mat1 and mat2 shapes cannot be multiplied (",
      mat1_sizes[0],
      "x",
      mat1_sizes[1],
      " and ",
      mat2_sizes[0],
      "x",
      mat2_sizes[1],
      ")");

  IntArrayRef self_sizes = self.sizes();
  TORCH_CHECK(
      self_sizes[0] == mat1_sizes[0], "sampled_addmm: self dim 0 must match mat1 dim 0");
  TORCH_CHECK(
      self_sizes[1] == mat2_sizes[1], "sampled_addmm: self dim 1 must match mat2 dim 1");

  if (&result != &self) {
    at::native::resize_as_sparse_csr_(result, self);
    result.copy_(self);
  }

  // there's a segfault when calling cuSPARSE on 0-sized matrices
  if (mat1.numel() == 0 || mat2.numel() == 0) {
    return result;
  }

  sparse::impl::cuda::sampled_addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
  return result;
}

Tensor sparse_sampled_addmm_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  auto result = at::empty({0, 0}, self.options());
  at::native::sparse_sampled_addmm_out_sparse_csr_cuda(self, mat1, mat2, beta, alpha, result);
  return result;
}

Tensor& addmm_out_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());

  // Same checks as in TORCH_META_FUNC(addmm) at
  // aten/src/ATen/native/LinearAlgebra.cpp
  TORCH_CHECK(
      mat1.dim() == 2, "mat1 must be a matrix, got ", mat1.dim(), "-D tensor");
  TORCH_CHECK(
      mat2.dim() == 2, "mat2 must be a matrix, got ", mat2.dim(), "-D tensor");

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(
      mat1_sizes[1] == mat2_sizes[0],
      "mat1 and mat2 shapes cannot be multiplied (",
      mat1_sizes[0],
      "x",
      mat1_sizes[1],
      " and ",
      mat2_sizes[0],
      "x",
      mat2_sizes[1],
      ")");

  // From addmm_out_cuda_impl at ATen/native/cuda/Blas.cpp
  // TODO: remove code duplication and unify code
  // There were undefined symbol problems,
  // when using the same function for CUDA and SparseCsrCUDA dispatch keys
  // Also structured kernels do not support sparse output
  IntArrayRef self__sizes;
  c10::MaybeOwned<Tensor> self_;
  if (&result != &self && self.layout() == kStrided) {
    self_ = expand_size(self, {mat1_sizes[0], mat2_sizes[1]}, "addmm");
    self__sizes = self_->sizes();
  } else {
    self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    self__sizes = self_->sizes();
    TORCH_CHECK(result.dim() == 2, "tensors must be 2-D");
    TORCH_CHECK(
        self__sizes[0] == mat1_sizes[0], "self_ dim 0 must match mat1 dim 0");
    TORCH_CHECK(
        self__sizes[1] == mat2_sizes[1], "self_ dim 1 must match mat2 dim 1");
  }

  if (&result != &self) {
    if (result.layout() == kStrided) {
      at::native::resize_output(result, self__sizes);
    } else {
      at::native::resize_as_sparse_csr_(result, *self_);
    }
    result.copy_(*self_);
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  if (mat1._nnz() == 0 && mat2.layout() == kStrided) {
    // According to docs, when beta==0 values in self should be ignored
    // nans and infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      result.zero_();
    } else {
      result.mul_(beta);
    }
    return result;
  }

  if (mat2.is_sparse_csr() && (mat1._nnz() == 0 || mat2._nnz() == 0)) {
    if (beta.toComplexDouble() == 0.) {
      result.values().zero_();
    } else {
      result.values().mul_(beta);
    }
    return result;
  }

  sparse::impl::cuda::addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
  return result;
}

Tensor& addmv_out_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.is_sparse_csr());

  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  // Preprocessing code is copied from TORCH_IMPL_FUNC(addmv_out_cuda) at
  // aten/src/ATen/native/cuda/Blas.cpp
  // It would be nice to have it unified but there were undefined symbol
  // problems, when using the same function for CUDA and SparseCsrCUDA dispatch
  // keys and structured kernel
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
              c10::nullopt /* layout */,
              at::kCPU,
              c10::nullopt /* pin_memory */));
    }
  }

  sparse::impl::cuda::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
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
std::tuple<Tensor&, Tensor&> triangular_solve_out_sparse_csr_cuda(
    const Tensor& B,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    Tensor& X,
    Tensor& clone_A) {
  sparse::impl::cuda::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
  return std::tuple<Tensor&, Tensor&>(X, clone_A);
}

} // namespace native
} // namespace at
