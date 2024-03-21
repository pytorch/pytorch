#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mkl/SparseCsrLinearAlgebra.h>
#include <ATen/native/SparseTensorUtils.h>

// Don't compile with MKL for macos since linking the sparse MKL routines
// needs some build fixes.
// Macros source:
// https://web.archive.org/web/20191012035921/http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
#if !AT_MKL_ENABLED() || defined(__APPLE__) || \
    defined(__MACH__)

namespace at {
namespace sparse_csr {
Tensor& _sparse_mm_mkl_(
    Tensor& self,
    const SparseCsrTensor& sparse_,
    const Tensor& dense,
    const Tensor& t,
    const Scalar& alpha,
    const Scalar& beta) {
#if __APPLE__ || __MACH__
  AT_ERROR("sparse_mm_mkl: MKL support is disabled on macos/iOS.");
#else
  AT_ERROR("sparse_mm_mkl: ATen not compiled with MKL support");
#endif
  return self; // for stopping compiler warnings.
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

namespace at {
namespace sparse_csr {

#ifdef MKL_ILP64
static constexpr ScalarType TORCH_INT_TYPE = at::kLong;
#else
static constexpr ScalarType TORCH_INT_TYPE = at::kInt;
#endif

class SparseCsrMKLInterface {
 private:
  sparse_matrix_t A{nullptr};
  matrix_descr desc;

 public:
  SparseCsrMKLInterface(
      MKL_INT* col_indices,
      MKL_INT* crow_indices,
      double* values,
      MKL_INT nrows,
      MKL_INT ncols) {
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
        retval == 0,
        "mkl_sparse_d_create_csr failed with error code: ",
        retval);
  }

  SparseCsrMKLInterface(
      MKL_INT* col_indices,
      MKL_INT* crow_indices,
      float* values,
      MKL_INT nrows,
      MKL_INT ncols) {
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
        retval == 0,
        "mkl_sparse_s_create_csr failed with error code: ",
        retval);
  }

 // res(nrows, dense_ncols) = (sparse(nrows * ncols) @ dense(ncols x dense_ncols))
  inline void sparse_mm(
      float* res,
      float* dense,
      float alpha,
      float beta,
      MKL_INT nrows,
      MKL_INT ncols,
      MKL_INT dense_ncols) {
    int stat;
    if (dense_ncols == 1) {
      stat = mkl_sparse_s_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        dense,
        beta,
        res);
      TORCH_CHECK(stat == 0, "mkl_sparse_s_mv failed with error code: ", stat);
    } else {
      stat = mkl_sparse_s_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        SPARSE_LAYOUT_ROW_MAJOR,
        dense,
        nrows,
        ncols,
        beta,
        res,
        dense_ncols);
      TORCH_CHECK(stat == 0, "mkl_sparse_s_mm failed with error code: ", stat);
    }
  }

  inline void sparse_mm(
      double* res,
      double* dense,
      double alpha,
      double beta,
      MKL_INT nrows,
      MKL_INT ncols,
      MKL_INT dense_ncols) {
    int stat;
    if (dense_ncols == 1) {
      stat = mkl_sparse_d_mv(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        dense,
        beta,
        res);
      TORCH_CHECK(stat == 0, "mkl_sparse_d_mv failed with error code: ", stat);
    }
    else {
      stat = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        A,
        desc,
        SPARSE_LAYOUT_ROW_MAJOR,
        dense,
        nrows,
        ncols,
        beta,
        res,
        dense_ncols);
      TORCH_CHECK(stat == 0, "mkl_sparse_d_mm failed with error code: ", stat);
    }
  }

  ~SparseCsrMKLInterface() {
    mkl_sparse_destroy(A);
  }
};

template <typename scalar_t>
static inline void sparse_mm_mkl_template(
    Tensor& res,
    const Tensor& col_indices,
    const Tensor& crow_indices,
    const Tensor& values,
    const Tensor& dense,
    const Tensor& t,
    const Scalar& alpha,
    const Scalar& beta,
    IntArrayRef size,
    IntArrayRef dense_size) {
  SparseCsrMKLInterface mkl_impl(
      col_indices.data_ptr<MKL_INT>(),
      crow_indices.data_ptr<MKL_INT>(),
      values.data_ptr<scalar_t>(),
      size[0],
      size[1]);
  mkl_impl.sparse_mm(
      res.data_ptr<scalar_t>(),
      dense.data_ptr<scalar_t>(),
      alpha.to<scalar_t>(),
      beta.to<scalar_t>(),
      size[0],
      size[1],
      dense_size[1]);
}

static bool inline constexpr is_mkl_int32_index() {
#ifdef MKL_ILP64
  return false;
#else
  return true;
#endif
}

Tensor& _sparse_mm_mkl_(
    Tensor& self,
    const SparseCsrTensor& sparse_,
    const Tensor& dense,
    const Tensor& t,
    const Scalar& alpha,
    const Scalar& beta) {
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
