#include <TH/THBlas.h>
#include <ATen/ScalarType.h>

// This header file shouldn't be anything permanent; it's just a temporary
// dumping ground to help you get access to utilities in THBlas.h via templates,
// rather than by name directly.  Someone should figure out a reasonable way to
// rewrite these in more idiomatic ATen and move it into ATen proper.

#define AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(_) \
_(uint8_t,Byte,i) \
_(int8_t,Char,i) \
_(int16_t,Short,i) \
_(int,Int,i) \
_(int64_t,Long,i) \
_(float,Float,d) \
_(double,Double,d)

template<typename T>
inline void THBlas_axpy(int64_t n, T a, T *x, int64_t incx, T *y, int64_t incy);

#define AXPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_axpy<ctype>(int64_t n, ctype a, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_axpy(n, a, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(AXPY_SPECIALIZATION)


template<typename T>
inline void THBlas_copy(int64_t n, T *x, int64_t incx, T *y, int64_t incy);

#define COPY_SPECIALIZATION(ctype,name,_1) \
  template<> \
  inline void THBlas_copy<ctype>(int64_t n, ctype *x, int64_t incx, \
                   ctype *y, int64_t incy) { \
    TH ## name ## Blas_copy(n, x, incx, y, incy); \
  }

AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(COPY_SPECIALIZATION)
