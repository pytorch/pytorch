#include <ATen/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/LinearAlgebraUtils.h>
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

c10::MaybeOwned<Tensor> inline prepare_dense_matrix_for_mkl(
    const Tensor& tensor) {
  if (tensor.is_non_overlapping_and_dense() ||
      is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

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

void triangular_solve_out_sparse_csr(
    const Tensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_USE_MKL_SPARSE()
  TORCH_CHECK(
      false,
      "Calling triangular_solve on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  if (B.numel() == 0 || X.numel() == 0 || A._nnz() == 0) {
    return;
  }

  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_mkl(B);
  c10::MaybeOwned<Tensor> X_ = prepare_dense_matrix_for_mkl(X);

  sparse_operation_t opA = transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descrA.mode = upper ? SPARSE_FILL_MODE_UPPER : SPARSE_FILL_MODE_LOWER;
  descrA.diag = unitriangular ? SPARSE_DIAG_UNIT : SPARSE_DIAG_NON_UNIT;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      X.scalar_type(), "triangular_solve_out_sparse_csr_impl_mkl", [&] {
        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(A);
        scalar_t alpha = 1;

        if (B.size(-1) == 1) {
          at::mkl::sparse::trsv<scalar_t>(
              opA,
              alpha,
              mkl_sparse_mat.descriptor(),
              descrA,
              B_->data_ptr<scalar_t>(),
              X_->data_ptr<scalar_t>());
        } else {
          // TODO: support mixed memory format
          IntArrayRef X_strides = X_->strides();
          IntArrayRef B_strides = B_->strides();
          auto ndim = X_->dim();
          bool is_X_row_major = (X_strides[ndim - 1] == 1);
          bool is_B_row_major = (B_strides[ndim - 1] == 1);
          TORCH_INTERNAL_ASSERT(is_X_row_major && is_B_row_major);

          auto order = is_X_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
          auto nrhs = mkl_int_cast(B.size(-1), "nrhs");
          auto ldx = is_X_row_major ? X_strides[ndim - 2] : X_strides[ndim - 1];
          auto ldb = is_B_row_major ? B_strides[ndim - 2] : B_strides[ndim - 1];
          at::mkl::sparse::trsm<scalar_t>(
              opA,
              alpha,
              mkl_sparse_mat.descriptor(),
              descrA,
              order,
              B_->data_ptr<scalar_t>(),
              nrhs,
              ldb,
              X_->data_ptr<scalar_t>(),
              ldx);
        }
      });

  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif
}

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
