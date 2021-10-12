#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>

#include <c10/util/MaybeOwned.h>

namespace at {
namespace native {

Tensor& addmm_out_sparse_csr_dense_cuda(
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
  if (&result != &self) {
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
    at::native::resize_output(result, self__sizes);
    if (beta.toComplexDouble() != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  IntArrayRef result_sizes = result.sizes();
  if ((result_sizes[0] == 0) || (result_sizes[1] == 0)) {
    return result;
  }

  if (mat1._nnz() == 0) {
    // By definition, when beta==0, values in self should be ignored. nans and
    // infs should not propagate
    if (beta.toComplexDouble() == 0.) {
      return result.zero_();
    }
    return at::mul_out(
        result,
        self,
        at::native::scalar_tensor(
            beta,
            self.scalar_type(),
            c10::nullopt /* layout */,
            at::kCPU,
            c10::nullopt /* pin_memory */));
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

} // namespace native
} // namespace at
