/*
  Provides the implementations of cuSPARSE function templates.
*/

#include <ATen/cuda/CUDASparseBlas.h>

namespace at::cuda::sparse {

template <>
void csrgeam2_bufferSizeExt<float>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseScsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<double>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const cuComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2_bufferSizeExt<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_BUFFERSIZE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZcsrgeam2_bufferSizeExt(
      handle,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBufferSizeInBytes));
}

template <>
void csrgeam2<float>(CUSPARSE_CSRGEAM2_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseScsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<double>(CUSPARSE_CSRGEAM2_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDcsrgeam2(
      handle,
      m,
      n,
      alpha,
      descrA,
      nnzA,
      csrSortedValA,
      csrSortedRowPtrA,
      csrSortedColIndA,
      beta,
      descrB,
      nnzB,
      csrSortedValB,
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      csrSortedValC,
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<c10::complex<float>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<cuComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void csrgeam2<c10::complex<double>>(
    CUSPARSE_CSRGEAM2_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZcsrgeam2(
      handle,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      nnzA,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValA),
      csrSortedRowPtrA,
      csrSortedColIndA,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      descrB,
      nnzB,
      reinterpret_cast<const cuDoubleComplex*>(csrSortedValB),
      csrSortedRowPtrB,
      csrSortedColIndB,
      descrC,
      reinterpret_cast<cuDoubleComplex*>(csrSortedValC),
      csrSortedRowPtrC,
      csrSortedColIndC,
      pBuffer));
}

template <>
void bsrmm<float>(CUSPARSE_BSRMM_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}

template <>
void bsrmm<double>(CUSPARSE_BSRMM_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      B,
      ldb,
      beta,
      C,
      ldc));
}

template <>
void bsrmm<c10::complex<float>>(CUSPARSE_BSRMM_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuComplex*>(B),
      ldb,
      reinterpret_cast<const cuComplex*>(beta),
      reinterpret_cast<cuComplex*>(C),
      ldc));
}

template <>
void bsrmm<c10::complex<double>>(
    CUSPARSE_BSRMM_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrmm(
      handle,
      dirA,
      transA,
      transB,
      mb,
      n,
      kb,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuDoubleComplex*>(B),
      ldb,
      reinterpret_cast<const cuDoubleComplex*>(beta),
      reinterpret_cast<cuDoubleComplex*>(C),
      ldc));
}

template <>
void bsrmv<float>(CUSPARSE_BSRMV_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<double>(CUSPARSE_BSRMV_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      x,
      beta,
      y));
}

template <>
void bsrmv<c10::complex<float>>(CUSPARSE_BSRMV_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuComplex*>(x),
      reinterpret_cast<const cuComplex*>(beta),
      reinterpret_cast<cuComplex*>(y)));
}

template <>
void bsrmv<c10::complex<double>>(
    CUSPARSE_BSRMV_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrmv(
      handle,
      dirA,
      transA,
      mb,
      nb,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      reinterpret_cast<const cuDoubleComplex*>(x),
      reinterpret_cast<const cuDoubleComplex*>(beta),
      reinterpret_cast<cuDoubleComplex*>(y)));
}

#if AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()

template <>
void bsrsv2_bufferSize<float>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<double>(CUSPARSE_BSRSV2_BUFFER_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSV2_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_bufferSize(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsv2_analysis<float>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<double>(CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSV2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_analysis(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<float>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      x,
      y,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<double>(CUSPARSE_BSRSV2_SOLVE_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      x,
      y,
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<c10::complex<float>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const cuComplex*>(x),
      reinterpret_cast<cuComplex*>(y),
      policy,
      pBuffer));
}

template <>
void bsrsv2_solve<c10::complex<double>>(
    CUSPARSE_BSRSV2_SOLVE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsv2_solve(
      handle,
      dirA,
      transA,
      mb,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const cuDoubleComplex*>(x),
      reinterpret_cast<cuDoubleComplex*>(y),
      policy,
      pBuffer));
}

template <>
void bsrsm2_bufferSize<float>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<double>(CUSPARSE_BSRSM2_BUFFER_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<c10::complex<float>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_bufferSize<c10::complex<double>>(
    CUSPARSE_BSRSM2_BUFFER_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_bufferSize(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      pBufferSizeInBytes));
}

template <>
void bsrsm2_analysis<float>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<double>(CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<float>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_analysis<c10::complex<double>>(
    CUSPARSE_BSRSM2_ANALYSIS_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_analysis(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<float>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(float)) {
  TORCH_CUDASPARSE_CHECK(cusparseSbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      B,
      ldb,
      X,
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<double>(CUSPARSE_BSRSM2_SOLVE_ARGTYPES(double)) {
  TORCH_CUDASPARSE_CHECK(cusparseDbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      alpha,
      descrA,
      bsrValA,
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      B,
      ldb,
      X,
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<float>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<float>)) {
  TORCH_CUDASPARSE_CHECK(cusparseCbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      reinterpret_cast<const cuComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const cuComplex*>(B),
      ldb,
      reinterpret_cast<cuComplex*>(X),
      ldx,
      policy,
      pBuffer));
}

template <>
void bsrsm2_solve<c10::complex<double>>(
    CUSPARSE_BSRSM2_SOLVE_ARGTYPES(c10::complex<double>)) {
  TORCH_CUDASPARSE_CHECK(cusparseZbsrsm2_solve(
      handle,
      dirA,
      transA,
      transX,
      mb,
      n,
      nnzb,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      descrA,
      reinterpret_cast<const cuDoubleComplex*>(bsrValA),
      bsrRowPtrA,
      bsrColIndA,
      blockDim,
      info,
      reinterpret_cast<const cuDoubleComplex*>(B),
      ldb,
      reinterpret_cast<cuDoubleComplex*>(X),
      ldx,
      policy,
      pBuffer));
}

#endif // AT_USE_HIPSPARSE_TRIANGULAR_SOLVE

} // namespace at::cuda::sparse
