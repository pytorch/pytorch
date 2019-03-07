#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THBlas.h"
#else

/* Level 1 */
TH_API void THBlas_(swap)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
TH_API void THBlas_(scal)(int64_t n, scalar_t a, scalar_t *x, int64_t incx);
TH_API void THBlas_(copy)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
TH_API void THBlas_(axpy)(int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);
TH_API scalar_t THBlas_(dot)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy);

/* Level 2 */
TH_API void THBlas_(gemv)(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);
TH_API void THBlas_(ger)(int64_t m, int64_t n, scalar_t alpha, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy, scalar_t *a, int64_t lda);

/* Level 3 */
TH_API void THBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *b, int64_t ldb, scalar_t beta, scalar_t *c, int64_t ldc);

#endif
