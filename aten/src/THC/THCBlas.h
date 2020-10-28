#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <c10/util/BFloat16.h>

/* Level 2 */
THC_API void THCudaBlas_Sger(THCState *state, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda);
THC_API void THCudaBlas_Dger(THCState *state, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda);

/* Level 3 */
THC_API void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc);
THC_API void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc);

THC_API void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, THHalf alpha, THHalf *a, int64_t lda, THHalf *b, int64_t ldb, THHalf beta, THHalf *c, int64_t ldc);

THC_API void THCudaBlas_Bgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::BFloat16 alpha, at::BFloat16 *a, int64_t lda, at::BFloat16 *b, int64_t ldb, at::BFloat16 beta, at::BFloat16 *c, int64_t ldc);

THC_API void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     float alpha, const float *a[], int64_t lda, const float *b[], int64_t ldb,
                                     float beta, float *c[], int64_t ldc, int64_t batchCount);
THC_API void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     double alpha, const double *a[], int64_t lda, const double *b[], int64_t ldb,
                                     double beta, double *c[], int64_t ldc, int64_t batchCount);
THC_API void THCudaBlas_SgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     float alpha, const float *a, int64_t lda, int64_t strideA, const float *b, int64_t ldb, int64_t strideB,
                                     float beta, float *c, int64_t ldc, int64_t strideC, int64_t batchCount);
THC_API void THCudaBlas_DgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     double alpha, const double *a, int64_t lda, int64_t strideA, const double *b, int64_t ldb, int64_t strideB,
                                     double beta, double *c, int64_t ldc, int64_t strideC, int64_t batchCount);

void THCudaBlas_HgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     THHalf alpha, const THHalf *a, int64_t lda, int64_t strideA, const THHalf *b, int64_t ldb, int64_t strideB,
                                                                  THHalf beta, THHalf *c, int64_t ldc, int64_t strideC, int64_t batchCount);

void THCudaBlas_BgemmStridedBatched(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k,
                                     at::BFloat16 alpha, const at::BFloat16 *a, int64_t lda, int64_t strideA, const at::BFloat16 *b, int64_t ldb, int64_t strideB,
                                     at::BFloat16 beta, at::BFloat16 *c, int64_t ldc, int64_t strideC, int64_t batchCount);

#endif
