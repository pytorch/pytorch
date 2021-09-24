#pragma once

/*
  Provides a subset of MKL Sparse BLAS functions as templates:

    mv<scalar_t>(operation, alpha, A, descr, x, beta, y)

  where scalar_t is double, float, c10::complex<double> or c10::complex<float>.
  The functions are available in at::mkl::sparse namespace.
*/

#include <c10/util/complex.h>
#include <c10/util/Exception.h>

#include <mkl_spblas.h>

namespace at {
namespace mkl {
namespace sparse {

#define MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)                              \
  sparse_matrix_t* A, const sparse_index_base_t indexing, const MKL_INT rows, \
      const MKL_INT cols, MKL_INT* rows_start, MKL_INT* rows_end,             \
      MKL_INT* col_indx, scalar_t* values

template <typename scalar_t>
inline void create_csr(MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::create_csr: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void create_csr<float>(MKL_SPARSE_CREATE_CSR_ARGTYPES(float));
template <>
void create_csr<double>(MKL_SPARSE_CREATE_CSR_ARGTYPES(double));
template <>
void create_csr<c10::complex<float>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<float>));
template <>
void create_csr<c10::complex<double>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<double>));

#define MKL_SPARSE_MV_ARGTYPES(scalar_t)                        \
  const sparse_operation_t operation, const scalar_t alpha,     \
      const sparse_matrix_t A, const struct matrix_descr descr, \
      const scalar_t* x, const scalar_t beta, scalar_t* y

template <typename scalar_t>
inline void mv(MKL_SPARSE_MV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::mv: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void mv<float>(MKL_SPARSE_MV_ARGTYPES(float));
template <>
void mv<double>(MKL_SPARSE_MV_ARGTYPES(double));
template <>
void mv<c10::complex<float>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<float>));
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>));

} // namespace sparse
} // namespace mkl
} // namespace at
