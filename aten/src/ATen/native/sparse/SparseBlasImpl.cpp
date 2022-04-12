#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Config.h>
#include <ATen/native/mkl/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlasImpl.h>

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace cpu {

/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr(
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
  Computes a sum of two sparse matrices defined as
  result <- mat1 + alpha*mat2

  Args:
  * `mat1` - CSR Tensor storing sparse m x n matrix.
  * `mat2` - CSR Tensor storing sparse m x n matrix.
  * `result` - [in] CSR Tensor storing sparse m x n matrix.
               [out] result of the operation.
*/
void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  sparse::impl::mkl::add_out_sparse_csr(mat1, mat2, alpha, result);
#endif
}

void triangular_solve_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_MKL_ENABLED()
  TORCH_CHECK(
      false,
      "Calling triangular_solve on a sparse CPU tensor requires compiling PyTorch with MKL. ",
      "Please use PyTorch built MKL support.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.is_sparse_csr());
  sparse::impl::mkl::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
#endif
}

} // namespace cpu
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
