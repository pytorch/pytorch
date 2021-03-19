#include <ATen/ATen.h>
#include <ATen/SparseCsrTensorUtils.h>

// Don't compile with MKL for VS code since linking the sparse MKL routines
// needs some build fixes.
// https://github.com/pytorch/pytorch/pull/50937#issuecomment-778732740
#if !AT_MKL_ENABLED() || _MSC_VER

namespace at {
namespace native {
using namespace at::sparse;
TORCH_API Tensor _sparse_mm_mkl_(
    Tensor& self,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Tensor& t,
    Scalar alpha,
    Scalar beta) {
#if _MSC_VER
  AT_ERROR("sparse_mm_mkl: MKL support is disabled on Windows");
#else
  AT_ERROR("sparse_mm_mkl: ATen not compiled with MKL support");
#endif
}
} // namespace native
} // namespace at

#else // AT_MKL_ENABLED

#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Limits.h>
#include <mkl.h>
#include <mkl_spblas.h>

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorImpl.h>

#ifdef MKL_ILP64
#define TORCH_INT_TYPE at::kLong
#else
#define TORCH_INT_TYPE at::kInt
#endif

namespace at {
namespace native {
using namespace at::sparse;

static inline void sparse_mm_mkl_impl(
    float* res,
    MKL_INT* col_indices,
    MKL_INT* crow_indices,
    float* values,
    float* dense,
    float* t,
    float alpha,
    float beta,
    MKL_INT nrows,
    MKL_INT ncols,
    MKL_INT dense_ncols) {
  sparse_matrix_t A = 0;
  matrix_descr desc;
  desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  int retval = mkl_sparse_s_create_csr(
      &A,
      SPARSE_INDEX_BASE_ZERO,
      nrows,
      ncols,
      crow_indices,
      crow_indices + 1,
      col_indices,
      values);
  TORCH_CHECK(
      retval == 0, "mkl_sparse_s_create_csr failed with error code: ", retval);

  mkl_sparse_s_mm(
      SPARSE_OPERATION_NON_TRANSPOSE,
      alpha,
      A,
      desc,
      SPARSE_LAYOUT_ROW_MAJOR,
      dense,
      dense_ncols,
      dense_ncols,
      beta,
      res,
      dense_ncols);
  mkl_sparse_destroy(A);
}

static inline void sparse_mm_mkl_impl(
    double* res,
    MKL_INT* col_indices,
    MKL_INT* crow_indices,
    double* values,
    double* dense,
    double* t,
    double alpha,
    double beta,
    MKL_INT nrows,
    MKL_INT ncols,
    MKL_INT dense_ncols) {
  sparse_matrix_t A = 0;
  matrix_descr desc;
  desc.type = SPARSE_MATRIX_TYPE_GENERAL;
  int retval = mkl_sparse_d_create_csr(
      &A,
      SPARSE_INDEX_BASE_ZERO,
      nrows,
      ncols,
      crow_indices,
      crow_indices + 1,
      col_indices,
      values);
  TORCH_CHECK(
      retval == 0, "mkl_sparse_d_create_csr failed with error code: ", retval);

  mkl_sparse_d_mm(
      SPARSE_OPERATION_NON_TRANSPOSE,
      alpha,
      A,
      desc,
      SPARSE_LAYOUT_ROW_MAJOR,
      dense,
      dense_ncols,
      dense_ncols,
      beta,
      res,
      dense_ncols);
  mkl_sparse_destroy(A);
}

template <typename scalar_t>
static inline void sparse_mm_mkl_template(
    Tensor& res,
    const Tensor& col_indices,
    const Tensor& crow_indices,
    const Tensor& values,
    const Tensor& dense,
    const Tensor& t,
    Scalar alpha,
    Scalar beta,
    IntArrayRef size,
    IntArrayRef dense_size) {
  sparse_mm_mkl_impl(
      res.data_ptr<scalar_t>(),
      col_indices.data_ptr<MKL_INT>(),
      crow_indices.data_ptr<MKL_INT>(),
      values.data_ptr<scalar_t>(),
      dense.data_ptr<scalar_t>(),
      t.data_ptr<scalar_t>(),
      alpha.to<scalar_t>(),
      beta.to<scalar_t>(),
      size[0],
      size[1],
      dense_size[1]);
}

static bool inline is_mkl_int32_index() {
#ifdef MKL_ILP64
  return false;
#else
  return true;
#endif
}

TORCH_API Tensor _sparse_mm_mkl_(
    Tensor& self,
    const SparseTensor& sparse_,
    const Tensor& dense,
    const Tensor& t,
    Scalar alpha,
    Scalar beta) {
  if (is_mkl_int32_index()) {
    if (sparse_.crow_indices().scalar_type() != kInt) {
      TORCH_WARN(
          "Pytorch is compiled with MKL LP64 and will convert crow_indices to int32.");
    }
    if (sparse_.col_indices().scalar_type() != kInt) {
      TORCH_WARN(
          "Pytorch is compiled with MKL LP64 and will convert col_indices to int32.");
    }
  } else { // This is for future proofing if we ever change to using MKL ILP64.
    if (sparse_.crow_indices().scalar_type() != kLong) {
      TORCH_WARN(
          "Pytorch is compiled with MKL ILP64 and will convert crow_indices dtype to int64.");
    }
    if (sparse_.col_indices().scalar_type() != kLong) {
      TORCH_WARN(
          "Pytorch is compiled with MKL ILP64 and will convert col_indices dtype to int64.");
    }
  }
  AT_DISPATCH_FLOATING_TYPES(
      dense.scalar_type(), "addmm_sparse_csr_dense", [&] {
        sparse_mm_mkl_template<scalar_t>(
            self,
            sparse_.col_indices().to(TORCH_INT_TYPE),
            sparse_.crow_indices().to(TORCH_INT_TYPE),
            sparse_.values(),
            dense,
            t,
            alpha,
            beta,
            sparse_.sizes(),
            dense.sizes());
      });
  return self;
}

} // namespace native
} // namespace at

#endif // AT_MKL_ENABLED
