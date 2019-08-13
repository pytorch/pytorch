#include <TH/THBlas.h>
#include <c10/core/ScalarType.h>

// This header file shouldn't be anything permanent; it's just a temporary
// dumping ground to help you get access to utilities in THBlas.h via templates,
// rather than by name directly.  Someone should figure out a reasonable way to
// rewrite these in more idiomatic ATen and move it into ATen proper.

template<typename T>
inline void THBlas_axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);

#define AXPY_SPECIALIZATION(ctype,name) \
  template<> \
  inline void THBlas_axpy<ctype>(int64_t n, ctype a, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_axpy(n, a, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES(AXPY_SPECIALIZATION)


template<typename T>
inline void THBlas_copy(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define COPY_SPECIALIZATION(ctype,name) \
  template<> \
  inline void THBlas_copy<ctype>(int64_t n, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_copy(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES(COPY_SPECIALIZATION)

template<typename T>
inline T THBlas_dot(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define DOT_SPECIALIZATION(ctype,name) \
  template<> \
  inline ctype THBlas_dot<ctype>(int64_t n, ctype *x, int64_t incx, ctype *y, int64_t incy) { \
    return TH ## name ## Blas_dot(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES(DOT_SPECIALIZATION)

template<typename T>
inline void THBlas_gemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    T alpha,
    T *a,
    int64_t lda,
    T *b,
    int64_t ldb,
    T beta,
    T *c,
    int64_t ldc);

#define GEMM_SPECIALIZATION(ctype,name) \
  template<> \
  inline void THBlas_gemm<ctype>( \
      char transa, \
      char transb, \
      int64_t m, \
      int64_t n, \
      int64_t k, \
      ctype alpha, \
      ctype *a, \
      int64_t lda, \
      ctype *b, \
      int64_t ldb, \
      ctype beta, \
      ctype *c, \
      int64_t ldc) { \
    TH ## name ## Blas_gemm(\
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); \
  }

AT_FORALL_SCALAR_TYPES(GEMM_SPECIALIZATION)

template <typename T>
inline void THBlas_gemv(
    char transa,
    int64_t m,
    int64_t n,
    T alpha,
    T* a,
    int64_t lda,
    T* x,
    int64_t incx,
    T beta,
    T* y,
    int64_t incy);

#define GEMV_SPECIALIZATION(ctype, name) \
  template <> \
  inline void THBlas_gemv<ctype>( \
      char transa, \
      int64_t m, \
      int64_t n, \
      ctype alpha, \
      ctype* a, \
      int64_t lda, \
      ctype* x, \
      int64_t incx, \
      ctype beta, \
      ctype* y, \
      int64_t incy) { \
    TH ## name ## Blas_gemv(transa, m, n, alpha, a, lda, x, incx, beta, y, incy); \
  }

 AT_FORALL_SCALAR_TYPES(GEMV_SPECIALIZATION)
