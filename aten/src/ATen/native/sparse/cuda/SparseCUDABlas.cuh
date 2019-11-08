#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>

namespace at { namespace native { namespace sparse { namespace cuda {

TORCH_CUDA_API void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr);

/* Level 3 */
TORCH_CUDA_API void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc);
TORCH_CUDA_API void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc);

// overloaded version
inline void csrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, float alpha, float *csrvala, int *csrrowptra, int *csrcolinda, float *b, int64_t ldb, float beta, float *c, int64_t ldc) { Scsrmm2(transa, transb, m, n, k, nnz, alpha, csrvala, csrrowptra, csrcolinda, b, ldb, beta, c, ldc); }
inline void csrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, double alpha, double *csrvala, int *csrrowptra, int *csrcolinda, double *b, int64_t ldb, double beta, double *c, int64_t ldc) { Dcsrmm2(transa, transb, m, n, k, nnz, alpha, csrvala, csrrowptra, csrcolinda, b, ldb, beta, c, ldc); }

/* format conversion */
TORCH_CUDA_API void CreateIdentityPermutation(int64_t nnz, int *P);
TORCH_CUDA_API void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes);
TORCH_CUDA_API void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer);
TORCH_CUDA_API void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes);
TORCH_CUDA_API void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer);

}}}} // namespace at::native::sparse::cuda
