/*
  Provides the implementations of MKL Sparse BLAS function templates.
*/

#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/SparseBlas.h>

namespace at {
namespace mkl {
namespace sparse {

const char* _mklGetErrorString(sparse_status_t error) {
  if (error == SPARSE_STATUS_SUCCESS) {
    return "SPARSE_STATUS_SUCCESS";
  }
  if (error == SPARSE_STATUS_NOT_INITIALIZED) {
    return "SPARSE_STATUS_NOT_INITIALIZED";
  }
  if (error == SPARSE_STATUS_ALLOC_FAILED) {
    return "SPARSE_STATUS_ALLOC_FAILED";
  }
  if (error == SPARSE_STATUS_INVALID_VALUE) {
    return "SPARSE_STATUS_INVALID_VALUE";
  }
  if (error == SPARSE_STATUS_EXECUTION_FAILED) {
    return "SPARSE_STATUS_EXECUTION_FAILED";
  }
  if (error == SPARSE_STATUS_INTERNAL_ERROR) {
    return "SPARSE_STATUS_INTERNAL_ERROR";
  }
  if (error == SPARSE_STATUS_NOT_SUPPORTED) {
    return "SPARSE_STATUS_NOT_SUPPORTED";
  }
  return "<unknown>";
}

template <>
void create_csr<float>(MKL_SPARSE_CREATE_CSR_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_create_csr(
      A, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}
template <>
void create_csr<double>(MKL_SPARSE_CREATE_CSR_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_create_csr(
      A, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}
template <>
void create_csr<c10::complex<float>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_create_csr(
      A,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex8*>(values)));
}
template <>
void create_csr<c10::complex<double>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_create_csr(
      A,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex16*>(values)));
}

template <>
void mv<float>(MKL_SPARSE_MV_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(
      mkl_sparse_s_mv(operation, alpha, A, descr, x, beta, y));
}
template <>
void mv<double>(MKL_SPARSE_MV_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(
      mkl_sparse_d_mv(operation, alpha, A, descr, x, beta, y));
}
template <>
void mv<c10::complex<float>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_mv(
      operation,
      reinterpret_cast<const MKL_Complex8&>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex8*>(x),
      reinterpret_cast<const MKL_Complex8&>(beta),
      reinterpret_cast<MKL_Complex8*>(y)));
}
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_mv(
      operation,
      reinterpret_cast<const MKL_Complex16&>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex16*>(x),
      reinterpret_cast<const MKL_Complex16&>(beta),
      reinterpret_cast<MKL_Complex16*>(y)));
}

template <>
void trsv<float>(MKL_SPARSE_TRSV_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_trsv(operation, alpha, A, descr, x, y));
}
template <>
void trsv<double>(MKL_SPARSE_TRSV_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_trsv(operation, alpha, A, descr, x, y));
}
template <>
void trsv<c10::complex<float>>(MKL_SPARSE_TRSV_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_trsv(
      operation,
      reinterpret_cast<const MKL_Complex8&>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex8*>(x),
      reinterpret_cast<MKL_Complex8*>(y)));
}
template <>
void trsv<c10::complex<double>>(
    MKL_SPARSE_TRSV_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_trsv(
      operation,
      reinterpret_cast<const MKL_Complex16&>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex16*>(x),
      reinterpret_cast<MKL_Complex16*>(y)));
}

template <>
void trsm<float>(MKL_SPARSE_TRSM_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_trsm(
      operation, alpha, A, descr, layout, x, columns, ldx, y, ldy));
}
template <>
void trsm<double>(MKL_SPARSE_TRSM_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_trsm(
      operation, alpha, A, descr, layout, x, columns, ldx, y, ldy));
}
template <>
void trsm<c10::complex<float>>(MKL_SPARSE_TRSM_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_trsm(
      operation,
      reinterpret_cast<const MKL_Complex8&>(alpha),
      A,
      descr,
      layout,
      reinterpret_cast<const MKL_Complex8*>(x),
      columns,
      ldx,
      reinterpret_cast<MKL_Complex8*>(y),
      ldy));
}
template <>
void trsm<c10::complex<double>>(
    MKL_SPARSE_TRSM_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_trsm(
      operation,
      reinterpret_cast<const MKL_Complex16&>(alpha),
      A,
      descr,
      layout,
      reinterpret_cast<const MKL_Complex16*>(x),
      columns,
      ldx,
      reinterpret_cast<MKL_Complex16*>(y),
      ldy));
}

} // namespace sparse
} // namespace mkl
} // namespace at
