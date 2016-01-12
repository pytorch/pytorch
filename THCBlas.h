#ifndef THC_BLAS_INC
#define THC_BLAS_INC

#include "THCGeneral.h"

/* Level 1 */
THC_API void THCudaBlas_swap(THCState *state, long n, float *x, long incx, float *y, long incy);
THC_API void THCudaBlas_scal(THCState *state, long n, float a, float *x, long incx);
THC_API void THCudaBlas_copy(THCState *state, long n, float *x, long incx, float *y, long incy);
THC_API void THCudaBlas_axpy(THCState *state, long n, float a, float *x, long incx, float *y, long incy);
THC_API float THCudaBlas_dot(THCState *state, long n, float *x, long incx, float *y, long incy);

/* Level 2 */
THC_API void THCudaBlas_gemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
THC_API void THCudaBlas_ger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);

/* Level 3 */
THC_API void THCudaBlas_gemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
THC_API void THCudaBlas_gemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                    float alpha, const float *a[], long lda, const float *b[], long ldb,
                                    float beta, float *c[], long ldc, long batchCount);

/* Inverse */
THC_API void THCudaBlas_getrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize);
THC_API void THCudaBlas_getri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize);

#endif
