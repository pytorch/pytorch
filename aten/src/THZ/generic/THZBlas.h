#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZBlas.h"
#else

/* Level 1 */
TH_API void THZBlas_(swap)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy);
TH_API void THZBlas_(scal)(int64_t n, ntype a, ntype *x, int64_t incx);
TH_API void THZBlas_(copy)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy);
TH_API void THZBlas_(axpy)(int64_t n, ntype a, ntype *x, int64_t incx, ntype *y, int64_t incy);
TH_API ntype THZBlas_(dot)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy);

/* Level 2 */
TH_API void THZBlas_(gemv)(char trans, int64_t m, int64_t n, ntype alpha, ntype *a, int64_t lda, ntype *x, int64_t incx, ntype beta, ntype *y, int64_t incy);
TH_API void THZBlas_(ger)(int64_t m, int64_t n, ntype alpha, ntype *x, int64_t incx, ntype *y, int64_t incy, ntype *a, int64_t lda);

/* Level 3 */
TH_API void THZBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, ntype alpha, ntype *a, int64_t lda, ntype *b, int64_t ldb, ntype beta, ntype *c, int64_t ldc);

#endif
