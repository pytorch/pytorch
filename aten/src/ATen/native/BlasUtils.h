#include <TH/THBlas.h>
#include <ATen/ScalarType.h>

// This header file shouldn't be anything permanent; it's just a temporary
// dumping ground to help you get access to utility in THBlas.h which aren't
// actually bound via ATen.  Someone should figure out a reasonable way to
// rewrite these in more idiomatic ATen.

namespace at { namespace thblas {

template<typename T>
inline void axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);

#define AXPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void axpy<ctype>(int64_t n, ctype a, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_axpy(n, a, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(AXPY_SPECIALIZATION)


template<typename T>
inline void copy(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define COPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void copy<ctype>(int64_t n, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_copy(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(COPY_SPECIALIZATION)

}} // thblas
