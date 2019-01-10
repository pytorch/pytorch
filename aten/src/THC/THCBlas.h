#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"
#include "THCHalf.h"

/* Level 1 */
THC_API float THCudaBlas_Sdot(THCState *state, int64_t n, float *x, int64_t incx, float *y, int64_t incy);
THC_API double THCudaBlas_Ddot(THCState *state, int64_t n, double *x, int64_t incx, double *y, int64_t incy);
#ifdef CUDA_HALF_TENSOR
THC_API half THCudaBlas_Hdot(THCState *state, int64_t n, half *x, int64_t incx, half *y, int64_t incy);
#endif

/* Level 2 */
THC_API void THCudaBlas_Sgemv(THCState *state, char trans, int64_t m, int64_t n, float alpha, float *a, int64_t lda, float *x, int64_t incx, float beta, float *y, int64_t incy);
THC_API void THCudaBlas_Dgemv(THCState *state, char trans, int64_t m, int64_t n, double alpha, double *a, int64_t lda, double *x, int64_t incx, double beta, double *y, int64_t incy);
THC_API void THCudaBlas_Sger(THCState *state, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda);
THC_API void THCudaBlas_Dger(THCState *state, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda);

/* Level 3 */
THC_API void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc);
THC_API void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc);

#ifdef CUDA_HALF_TENSOR
THC_API void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, half alpha, half *a, int64_t lda, half *b, int64_t ldb, half beta, half *c, int64_t ldc);
#endif

THC_API void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                                     float beta, float *c[], int64_t ldc, int64_t batchCount);
THC_API void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     double alpha, const double *a[], int64_t lda, const double *b[], int64_t ldb,
                                     double beta, double *c[], int64_t ldc, int64_t batchCount);
#if CUDA_VERSION >= 8000
THC_API void THCudaBlas_SgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     float alpha, const float *a, int64_t lda, int64_t strideA, const float *b, int64_t ldb, int64_t strideB,
                                     float beta, float *c, int64_t ldc, int64_t strideC, int64_t batchCount);
THC_API void THCudaBlas_DgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     double alpha, const double *a, int64_t lda, int64_t strideA, const double *b, int64_t ldb, int64_t strideB,
                                     double beta, double *c, int64_t ldc, int64_t strideC, int64_t batchCount);
#endif

#if CUDA_VERSION >= 9010
void THCudaBlas_HgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     half alpha, const half *a, int64_t lda, int64_t strideA, const half *b, int64_t ldb, int64_t strideB,
                                                                  half beta, half *c, int64_t ldc, int64_t strideC, int64_t batchCount);
#endif

/* Inverse */
THC_API void THCudaBlas_Sgetrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize);
THC_API void THCudaBlas_Dgetrf(THCState *state, int n, double **a, int lda, int *pivot, int *info, int batchSize);

THC_API void THCudaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize);
THC_API void THCudaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize);

THC_API void THCudaBlas_Sgetri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize);
THC_API void THCudaBlas_Dgetri(THCState *state, int n, const double **a, int lda, int *pivot, double **c, int ldc, int *info, int batchSize);

#endif
