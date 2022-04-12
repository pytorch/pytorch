#pragma once

/*
  Provides a subset of cuSPARSE functions as templates:

    csrgeam2<scalar_t>(...)

  where scalar_t is double, float, c10::complex<double> or c10::complex<float>.
  The functions are available in at::cuda::sparse namespace.
*/

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDASparse.h>

namespace at {
namespace cuda {
namespace sparse {

#define CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)             \
  cusparseHandle_t handle, int m, int n, const scalar_t *alpha,     \
      const cusparseMatDescr_t descrA, int nnzA,                    \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,   \
      const int *csrSortedColIndA, const scalar_t *beta,            \
      const cusparseMatDescr_t descrB, int nnzB,                    \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,   \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC, \
      const scalar_t *csrSortedValC, const int *csrSortedRowPtrC,   \
      const int *csrSortedColIndC, size_t *pBufferSizeInBytes

template <typename scalar_t>
inline void csrgeam2_bufferSizeExt(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::csrgeam2_bufferSizeExt: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void csrgeam2_bufferSizeExt<float>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float));
template <>
void csrgeam2_bufferSizeExt<double>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double));
template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>));
template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>));

#define CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()                                      \
  cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,     \
      int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA,     \
      const cusparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB, \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC,           \
      int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace

template <typename scalar_t>
inline void csrgeam2Nnz(CUSPARSE_CSRGEAM2_NNZ_ARGTYPES()) {
  TORCH_CUDASPARSE_CHECK(cusparseXcsrgeam2Nnz(
      handle,
      m,
      n,
      descrA,
      nnzA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      descrB,
      nnzB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedRowPtrC,
      nnzTotalDevHostPtr,
      workspace));
}

#define CUSPARSE_CSRGEAM2_ARGTYPES(scalar_t)                                 \
  cusparseHandle_t handle, int m, int n, const scalar_t *alpha,              \
      const cusparseMatDescr_t descrA, int nnzA,                             \
      const scalar_t *csrSortedValA, const int *csrSortedRowPtrA,            \
      const int *csrSortedColIndA, const scalar_t *beta,                     \
      const cusparseMatDescr_t descrB, int nnzB,                             \
      const scalar_t *csrSortedValB, const int *csrSortedRowPtrB,            \
      const int *csrSortedColIndB, const cusparseMatDescr_t descrC,          \
      scalar_t *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, \
      void *pBuffer

template <typename scalar_t>
inline void csrgeam2(CUSPARSE_CSRGEAM2_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::csrgeam2: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void csrgeam2<float>(CUSPARSE_CSRGEAM2_ARGTYPES(float));
template <>
void csrgeam2<double>(CUSPARSE_CSRGEAM2_ARGTYPES(double));
template <>
void csrgeam2<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>));
template <>
void csrgeam2<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRMM_ARGTYPES(scalar_t)                                    \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transB, int mb, int n, \
      int kb, int nnzb, const scalar_t *alpha,                               \
      const cusparseMatDescr_t descrA, const scalar_t *bsrValA,              \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      const scalar_t *B, int ldb, const scalar_t *beta, scalar_t *C, int ldc

template <typename scalar_t>
inline void bsrmm(CUSPARSE_BSRMM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrmm: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrmm<float>(CUSPARSE_BSRMM_ARGTYPES(float));
template <>
void bsrmm<double>(CUSPARSE_BSRMM_ARGTYPES(double));
template <>
void bsrmm<c10::complex<float>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<float>));
template <>
void bsrmm<c10::complex<double>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRMV_ARGTYPES(scalar_t)                                    \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, int mb, int nb, int nnzb,                  \
      const scalar_t *alpha, const cusparseMatDescr_t descrA,                \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, const scalar_t *x, const scalar_t *beta, scalar_t *y

template <typename scalar_t>
inline void bsrmv(CUSPARSE_BSRMV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrmv: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrmv<float>(CUSPARSE_BSRMV_ARGTYPES(float));
template <>
void bsrmv<double>(CUSPARSE_BSRMV_ARGTYPES(double));
template <>
void bsrmv<c10::complex<float>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<float>));
template <>
void bsrmv<c10::complex<double>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<double>));

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

#define CUSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)                 \
  cusparseHandle_t handle, cusparseDirection_t dirA,              \
      cusparseOperation_t transA, int mb, int nnzb,               \
      const cusparseMatDescr_t descrA, scalar_t *bsrValA,         \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim, \
      bsrsv2Info_t info, int *pBufferSizeInBytes

template <typename scalar_t>
inline void bsrsv2_bufferSize(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_bufferSize<float>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(float));
template <>
void bsrsv2_bufferSize<double>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(double));
template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(scalar_t)               \
  cusparseHandle_t handle, cusparseDirection_t dirA,              \
      cusparseOperation_t transA, int mb, int nnzb,               \
      const cusparseMatDescr_t descrA, const scalar_t *bsrValA,   \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim, \
      bsrsv2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsv2_analysis(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_analysis: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_analysis<float>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float));
template <>
void bsrsv2_analysis<double>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double));
template <>
void bsrsv2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)                           \
  cusparseHandle_t handle, cusparseDirection_t dirA,                       \
      cusparseOperation_t transA, int mb, int nnzb, const scalar_t *alpha, \
      const cusparseMatDescr_t descrA, const scalar_t *bsrValA,            \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,          \
      bsrsv2Info_t info, const scalar_t *x, scalar_t *y,                   \
      cusparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsv2_solve(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsv2_solve: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsv2_solve<float>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(float));
template <>
void bsrsv2_solve<double>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(double));
template <>
void bsrsv2_solve<c10::complex<float>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>));
template <>
void bsrsv2_solve<c10::complex<double>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)                            \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const cusparseMatDescr_t descrA, scalar_t *bsrValA,          \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, int *pBufferSizeInBytes

template <typename scalar_t>
inline void bsrsm2_bufferSize(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_bufferSize: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_bufferSize<float>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(float));
template <>
void bsrsm2_bufferSize<double>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(double));
template <>
void bsrsm2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)                          \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const cusparseMatDescr_t descrA, const scalar_t *bsrValA,    \
      const int *bsrRowPtrA, const int *bsrColIndA, int blockDim,            \
      bsrsm2Info_t info, cusparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsm2_analysis(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_analysis: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_analysis<float>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float));
template <>
void bsrsm2_analysis<double>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double));
template <>
void bsrsm2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>));

#define CUSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)                             \
  cusparseHandle_t handle, cusparseDirection_t dirA,                         \
      cusparseOperation_t transA, cusparseOperation_t transX, int mb, int n, \
      int nnzb, const scalar_t *alpha, const cusparseMatDescr_t descrA,      \
      const scalar_t *bsrValA, const int *bsrRowPtrA, const int *bsrColIndA, \
      int blockDim, bsrsm2Info_t info, const scalar_t *B, int ldb,           \
      scalar_t *X, int ldx, cusparseSolvePolicy_t policy, void *pBuffer

template <typename scalar_t>
inline void bsrsm2_solve(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::cuda::sparse::bsrsm2_solve: not implemented for ",
      typeid(scalar_t).name());
}

template <>
void bsrsm2_solve<float>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(float));
template <>
void bsrsm2_solve<double>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(double));
template <>
void bsrsm2_solve<c10::complex<float>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>));
template <>
void bsrsm2_solve<c10::complex<double>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>));

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

} // namespace sparse
} // namespace cuda
} // namespace at
