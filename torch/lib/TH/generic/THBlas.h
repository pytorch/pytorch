#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THBlas.h"
#else

/* Level 1 */
TH_API void THBlas_(swap)(int64_t n, real *x, int64_t incx, real *y, int64_t incy);
TH_API void THBlas_(scal)(int64_t n, real a, real *x, int64_t incx);
TH_API void THBlas_(copy)(int64_t n, real *x, int64_t incx, real *y, int64_t incy);
TH_API void THBlas_(axpy)(int64_t n, real a, real *x, int64_t incx, real *y, int64_t incy);
TH_API real THBlas_(dot)(int64_t n, real *x, int64_t incx, real *y, int64_t incy);

/* Level 2 */
TH_API void THBlas_(gemv)(char trans, int64_t m, int64_t n, real alpha, real *a, int64_t lda, real *x, int64_t incx, real beta, real *y, int64_t incy);
TH_API void THBlas_(ger)(int64_t m, int64_t n, real alpha, real *x, int64_t incx, real *y, int64_t incy, real *a, int64_t lda);

/* Level 3 */
TH_API void THBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, real alpha, real *a, int64_t lda, real *b, int64_t ldb, real beta, real *c, int64_t ldc);

#endif
