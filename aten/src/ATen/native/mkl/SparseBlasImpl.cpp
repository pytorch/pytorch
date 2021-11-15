#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/mkl/SparseBlasImpl.h>

#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>

#if AT_USE_MKL_SPARSE()
#include <ATen/mkl/SparseBlas.h>
#include <ATen/mkl/SparseDescriptors.h>
#endif

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace mkl {

namespace {

c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_mkl(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // anonymous namespace

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
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_mkl(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_mkl(vec);

  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmv_out_sparse_csr_impl_mkl", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat);

        at::mkl::sparse::mv<scalar_t>(
            opA,
            alpha_,
            mkl_sparse_mat.descriptor(),
            descrA,
            vec_->data_ptr<scalar_t>(),
            beta_,
            result_->data_ptr<scalar_t>());
      });

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
