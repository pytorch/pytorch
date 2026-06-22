#pragma once

#include <ATen/cuda/ATenCUDAGeneral.h>

namespace at::native::sparse::cuda{

TORCH_CUDA_CU_API void Xcoo2csr(
    const int* coorowind,
    int64_t nnz,
    int64_t m,
    int* csrrowptr);

/* Level 3 */
template <typename T>
TORCH_CUDA_CU_API void csrmm2(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t nnz,
    T alpha,
    T* csrvala,
    int* csrrowptra,
    int* csrcolinda,
    T* b,
    int64_t ldb,
    T beta,
    T* c,
    int64_t ldc);

/* format conversion */
TORCH_CUDA_CU_API void CreateIdentityPermutation(int64_t nnz, int* P);
TORCH_CUDA_CU_API void Xcsrsort_bufferSizeExt(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* csrRowPtr,
    const int* csrColInd,
    size_t* pBufferSizeInBytes);
TORCH_CUDA_CU_API void Xcsrsort(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* csrRowPtr,
    int* csrColInd,
    int* P,
    void* pBuffer);
TORCH_CUDA_CU_API void Xcoosort_bufferSizeExt(
    int64_t m,
    int64_t n,
    int64_t nnz,
    const int* cooRows,
    const int* cooCols,
    size_t* pBufferSizeInBytes);
TORCH_CUDA_CU_API void XcoosortByRow(
    int64_t m,
    int64_t n,
    int64_t nnz,
    int* cooRows,
    int* cooCols,
    int* P,
    void* pBuffer);
} // namespace at::native::sparse::cuda
