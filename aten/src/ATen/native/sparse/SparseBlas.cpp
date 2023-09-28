#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/SparseBlas.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/native/cpu/SampledAddmmKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <c10/util/MaybeOwned.h>

namespace at::native {

Tensor& addmv_out_sparse_compressed(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  TORCH_CHECK(
      mat.layout() != kSparseBsc,
      "torch.addmv: operation not supported for mat with SparseBsc layout");
  if (mat.layout() == kSparseCsc) {
    // TODO: Add native CSC support to avoid this expensive conversion
    return addmv_out_sparse_compressed(
        self, mat.to_sparse_csr(), vec, beta, alpha, result);
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mat.layout() == kSparseCsr || mat.layout() == kSparseBsr);

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

/*
  Computes `result` <- α*(A @ B) * spy(C) + β*C, where spy(C) is the sparsity pattern matrix of C.

  Args:
  * `mat1` - [in] dense Tensor A of size m × k.
  * `mat2` - [in] dense Tensor B of size k × n.
  * `self` - [in] sparse Tensor C of size m × n.
  * `result` - [out] sparse Tensor of size m × n.
*/
Tensor& sparse_sampled_addmm_out_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  at::native::sparse::sparse_sampled_addmm_check_inputs(self, mat1, mat2, beta, alpha, result);
  // Allow only same types as for the CUDA path
  auto t = self.scalar_type();
  TORCH_CHECK(t == ScalarType::Double || t == ScalarType::Float ||
    t == ScalarType::ComplexFloat || t == ScalarType::ComplexDouble,
    "sparse_sampled_addmm: Expected self to be a floating-point or complex tensor, but got ", t);
  if (&result != &self) {
    // We allow self to be a single matrix when mat1 and mat2 are batched
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(self._nnz(), result_sizes);
    result.copy_(self);
  }

  if (mat1.numel() == 0 || mat2.numel() == 0 || result._nnz() == 0) {
    result.mul_(beta);
    return result;
  }

  // transpose mat2 to [b, n, k] from performance perspective.
  // for gnn classic usage, mat2 is already stored in [b, n, k] physically,
  // so no extra memcpy is needed.
  auto mat2_t = mat2.transpose(-1, -2).contiguous();
  sampled_addmm_sparse_csr_stub(kCPU, mat1.contiguous(), mat2_t, beta, alpha, result);

  return result;
}

Tensor sparse_sampled_addmm_sparse_csr_cpu(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  auto result = at::empty({0, 0}, self.options());
  at::native::sparse_sampled_addmm_out_sparse_csr_cpu(self, mat1, mat2, beta, alpha, result);
  return result;
}

namespace sparse {

void sparse_sampled_addmm_check_inputs(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(self.is_sparse_csr());

  TORCH_CHECK(
      mat1.layout() == kStrided,
      "sampled_addmm: Expected mat1 to have strided layout, but got ",
      mat1.layout());
  TORCH_CHECK(
      mat2.layout() == kStrided,
      "sampled_addmm: Expected mat2 to have strided layout, but got ",
      mat2.layout());

  TORCH_CHECK(
      result.layout() == kSparseCsr,
      "sampled_addmm: Expected result to have sparse csr layout, but got ",
      result.layout());
  TORCH_CHECK(self.dense_dim() == 0,
      "sampled_addmm: Expected non-hybrid self tensor");
  TORCH_CHECK(result.dense_dim() == 0,
      "sampled_addmm: Expected non-hybrid result tensor");

  TORCH_CHECK(
      mat1.scalar_type() == mat2.scalar_type(),
      "sampled_addmm: Expected mat1 and mat2 to have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      mat2.scalar_type());
  TORCH_CHECK(
      mat1.scalar_type() == self.scalar_type(),
      "sampled_addmm: Expected mat1 and self to have the same dtype, but got ",
      mat1.scalar_type(),
      " and ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == self.scalar_type(),
      "sampled_addmm: Expected result and self to have the same dtype, but got ",
      result.scalar_type(),
      " and ",
      self.scalar_type());

  TORCH_CHECK(
      mat1.dim() >= 2,
      "sampled_addmm: Expected mat1 to be a matrix, got ",
      mat1.dim(),
      "-D tensor");
  TORCH_CHECK(
      mat2.dim() >= 2,
      "sampled_addmm: Expected mat2 to be a matrix, got ",
      mat2.dim(),
      "-D tensor");
  TORCH_CHECK(
      result.dim() >= 2,
      "sampled_addmm: Expected result to be a matrix, got ",
      result.dim(),
      "-D tensor");

  TORCH_CHECK(
    mat1.sizes().slice(0, mat1.dim() - 2) == mat2.sizes().slice(0, mat2.dim() - 2),
    "sampled_addmm: Expected mat1 and mat2 to have the same batch size, but got ",
    mat1.sizes().slice(0, mat1.dim() - 2),
    " and ",
    mat2.sizes().slice(0, mat2.dim() - 2));

  TORCH_CHECK(
    !(self.dim() > 2 && self.sizes().slice(0, self.dim() - 2) != mat1.sizes().slice(0, mat1.dim() - 2)),
    "sampled_addmm: Expected self and mat1 to have the same batch size, but got ",
    self.sizes().slice(0, self.dim() - 2),
    " and ",
    mat1.sizes().slice(0, mat1.dim() - 2));

  IntArrayRef mat1_sizes = mat1.sizes();
  IntArrayRef mat2_sizes = mat2.sizes();
  TORCH_CHECK(
      mat1_sizes[mat1.dim() - 1] == mat2_sizes[mat2.dim() - 2],
      "sampled_addmm: mat1 and mat2 shapes cannot be multiplied (",
      mat1_sizes[mat1.dim() - 2],
      "x",
      mat1_sizes[mat1.dim() - 1],
      " and ",
      mat2_sizes[mat2.dim() - 2],
      "x",
      mat2_sizes[mat2.dim() - 1],
      ")");

  IntArrayRef self_sizes = self.sizes();
  TORCH_CHECK(
      self_sizes[self.dim() - 2] == mat1_sizes[mat1.dim() - 2],
      "sampled_addmm: self.shape[-2] must match mat1.shape[-2]");
  TORCH_CHECK(
      self_sizes[self.dim() - 1] == mat2_sizes[mat2.dim() - 1],
      "sampled_addmm: self.shape[-1] must match mat2.shape[-1]");
}

} // namespace sparse

DEFINE_DISPATCH(sampled_addmm_sparse_csr_stub);

} // namespace at::native
