#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlas.h>

namespace at {
namespace native {

namespace {

/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr_cpu_impl(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
#endif
}

/*
  Solves a system of linear equations whose coefficients are represented in a sparse triangular matrix A:
  op(A) X = B.

  Args:
  * `A` - sparse Tensor of size m × m.
  * `B` - dense Tensor of size m × nrhs.
  * `X` - dense Tensor of size m × nrhs.
  * `upper` - controls whether upper or lower triangular part of A is considered in computations.
  * `transpose` - if true then op(A) = A^T.
  * `unitriangular` - if true then the diagonal elements of A are assumed to be one.
*/
void triangular_solve_out_sparse_csr_cpu_impl(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
  TORCH_INTERNAL_ASSERT(
      false,
      "Calling triangular solve on a sparse CPU tensor not implemented!");
}

} // anonymous namespace

REGISTER_ARCH_DISPATCH(addmv_out_sparse_csr_stub, DEFAULT, &addmv_out_sparse_csr_cpu_impl);
REGISTER_AVX_DISPATCH(addmv_out_sparse_csr_stub, &addmv_out_sparse_csr_cpu_impl);
REGISTER_AVX2_DISPATCH(addmv_out_sparse_csr_stub, &addmv_out_sparse_csr_cpu_impl);
REGISTER_VSX_DISPATCH(addmv_out_sparse_csr_stub, &addmv_out_sparse_csr_cpu_impl);

REGISTER_ARCH_DISPATCH(triangular_solve_out_sparse_csr_stub, DEFAULT, &triangular_solve_out_sparse_csr_cpu_impl);
REGISTER_AVX_DISPATCH(triangular_solve_out_sparse_csr_stub, &triangular_solve_out_sparse_csr_cpu_impl);
REGISTER_AVX2_DISPATCH(triangular_solve_out_sparse_csr_stub, &triangular_solve_out_sparse_csr_cpu_impl);
REGISTER_VSX_DISPATCH(triangular_solve_out_sparse_csr_stub, &triangular_solve_out_sparse_csr_cpu_impl);

} // namespace native
} // namespace at
