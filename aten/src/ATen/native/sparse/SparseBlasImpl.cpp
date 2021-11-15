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

} // namespace cpu
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
