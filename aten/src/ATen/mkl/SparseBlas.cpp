/*
  Provides the implementations of MKL Sparse BLAS function templates.
*/

#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/SparseBlas.h>

namespace at {
namespace mkl {
namespace sparse {

namespace {

  template <typename scalar_t, typename MKL_Complex>
  MKL_Complex to_mkl_complex(c10::complex<scalar_t> scalar) {
    MKL_Complex mkl_scalar;
    mkl_scalar.real = scalar.real();
    mkl_scalar.imag = scalar.imag();
    return mkl_scalar;
  }

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
      to_mkl_complex<float, MKL_Complex8>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex8*>(x),
      to_mkl_complex<float, MKL_Complex8>(beta),
      reinterpret_cast<MKL_Complex8*>(y)));
}
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_mv(
      operation,
      to_mkl_complex<double, MKL_Complex16>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex16*>(x),
      to_mkl_complex<double, MKL_Complex16>(beta),
      reinterpret_cast<MKL_Complex16*>(y)));
}

template <>
void add<float>(MKL_SPARSE_ADD_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_add(operation, A, alpha, B, C));
}
template <>
void add<double>(MKL_SPARSE_ADD_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_add(operation, A, alpha, B, C));
}
template <>
void add<c10::complex<float>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_add(
      operation, A, to_mkl_complex<float, MKL_Complex8>(alpha), B, C));
}
template <>
void add<c10::complex<double>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_add(
      operation, A, to_mkl_complex<double, MKL_Complex16>(alpha), B, C));
}

template <>
void export_csr<float>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_export_csr(
      source, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}
template <>
void export_csr<double>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_export_csr(
      source, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}
template <>
void export_csr<c10::complex<float>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_export_csr(
      source,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex8**>(values)));
}
template <>
void export_csr<c10::complex<double>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_export_csr(
      source,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex16**>(values)));
}

} // namespace sparse
} // namespace mkl
} // namespace at
