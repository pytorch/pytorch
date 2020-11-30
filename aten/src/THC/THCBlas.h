#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <c10/util/BFloat16.h>

/* Level 3 */
THC_API void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc);
THC_API void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc);

THC_API void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, THHalf alpha, THHalf *a, int64_t lda, THHalf *b, int64_t ldb, THHalf beta, THHalf *c, int64_t ldc);

THC_API void THCudaBlas_Bgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::BFloat16 alpha, at::BFloat16 *a, int64_t lda, at::BFloat16 *b, int64_t ldb, at::BFloat16 beta, at::BFloat16 *c, int64_t ldc);

#endif
