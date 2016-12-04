#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorMathSIMD.c"
#else

#ifndef _MSC_VER
#include <x86intrin.h>
#else
#include <intrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(TH_REAL_IS_DOUBLE)

void THTensor_(mul_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    #pragma omp parallel if(sz > TH_OMP_OVERHEAD_THRESHOLD)
    {
      #ifdef _OPENMP
      size_t num_threads = omp_get_num_threads();
      size_t tid = omp_get_thread_num();
      #else
      size_t num_threads = 1;
      size_t tid = 0;
      #endif
      ptrdiff_t i = tid * (sz / num_threads);
      ptrdiff_t i_end = tid == num_threads - 1 ? sz : i + sz / num_threads;
      __m256d YMM15 = _mm256_set_pd(value, value, value, value);
      __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
      for (; i<=((i_end)-8); i+=8) {
        YMM0 = _mm256_loadu_pd(tp+i);
        YMM1 = _mm256_loadu_pd(tp+i+4);
        YMM4 = _mm256_mul_pd(YMM0, YMM15);
        YMM5 = _mm256_mul_pd(YMM1, YMM15);
        _mm256_storeu_pd(rp+i, YMM4);
        _mm256_storeu_pd(rp+i+4, YMM5);
      }
      for (; i<i_end; i++) {
        rp[i] = tp[i] * value;
      }
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    #pragma omp parallel if(sz > TH_OMP_OVERHEAD_THRESHOLD)
    {
      #ifdef _OPENMP
      size_t num_threads = omp_get_num_threads();
      size_t tid = omp_get_thread_num();
      #else
      size_t num_threads = 1;
      size_t tid = 0;
      #endif
      ptrdiff_t i = tid * (sz / num_threads);
      ptrdiff_t i_end = tid == num_threads - 1 ? sz : i + sz / num_threads;
      __m256d YMM15 = _mm256_set_pd(value, value, value, value);
      __m256d YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7;
      for (; i<=((i_end)-8); i+=8) {
        YMM0 = _mm256_loadu_pd(tp+i);
        YMM1 = _mm256_loadu_pd(tp+i+4);
        YMM4 = _mm256_div_pd(YMM0, YMM15);
        YMM5 = _mm256_div_pd(YMM1, YMM15);
        _mm256_storeu_pd(rp+i, YMM4);
        _mm256_storeu_pd(rp+i+4, YMM5);
      }
      for (; i<i_end; i++) {
        rp[i] = tp[i] / value;
      }
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

#endif

#if defined(TH_REAL_IS_FLOAT)

void THTensor_(add_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *rp = THTensor_(data)(r_);
    real *tp = THTensor_(data)(t);
    ptrdiff_t sz = THTensor_(nElement)(t);
    #pragma omp parallel if(sz > TH_OMP_OVERHEAD_THRESHOLD)
    {
      #ifdef _OPENMP
      size_t num_threads = omp_get_num_threads();
      size_t tid = omp_get_thread_num();
      #else
      size_t num_threads = 1;
      size_t tid = 0;
      #endif
      ptrdiff_t i = tid * (sz / num_threads);
      ptrdiff_t i_end = tid == num_threads - 1 ? sz : i + sz / num_threads;
      __m256 YMM15 = _mm256_set_ps(value, value, value, value, value, value, value, value);
      __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7;
      for (; i<=((i_end)-16); i+=16) {
        YMM0 = _mm256_loadu_ps(tp+i);
        YMM1 = _mm256_loadu_ps(tp+i+8);
        YMM4 = _mm256_add_ps(YMM0, YMM15);
        YMM5 = _mm256_add_ps(YMM1, YMM15);
        _mm256_storeu_ps(rp+i, YMM4);
        _mm256_storeu_ps(rp+i+8, YMM5);
      }
      for (; i<i_end; i++) {
        rp[i] = tp[i] + value;
      }
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THTensor_(mul_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    #pragma omp parallel if(sz > TH_OMP_OVERHEAD_THRESHOLD)
    {
      #ifdef _OPENMP
      size_t num_threads = omp_get_num_threads();
      size_t tid = omp_get_thread_num();
      #else
      size_t num_threads = 1;
      size_t tid = 0;
      #endif
      ptrdiff_t i = tid * (sz / num_threads);
      ptrdiff_t i_end = tid == num_threads - 1 ? sz : i + sz / num_threads;
      __m256 YMM15 = _mm256_set_ps(value, value, value, value, value, value, value, value);
      __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5, YMM6, YMM7;
      for (; i<=((i_end)-16); i+=16) {
        YMM0 = _mm256_loadu_ps(tp+i);
        YMM1 = _mm256_loadu_ps(tp+i+8);
        YMM4 = _mm256_mul_ps(YMM0, YMM15);
        YMM5 = _mm256_mul_ps(YMM1, YMM15);
        _mm256_storeu_ps(rp+i, YMM4);
        _mm256_storeu_ps(rp+i+8, YMM5);
      }
      for (; i<i_end; i++) {
        rp[i] = tp[i] * value;
      }
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THTensor_(div_AVX)(THTensor *r_, THTensor *t, real value)
{
  THTensor_(resizeAs)(r_, t);
  if (THTensor_(isContiguous)(r_) && THTensor_(isContiguous)(t) && THTensor_(nElement)(r_) == THTensor_(nElement)(t)) {
    real *tp = THTensor_(data)(t);
    real *rp = THTensor_(data)(r_);
    ptrdiff_t sz = THTensor_(nElement)(t);
    #pragma omp parallel if(sz > TH_OMP_OVERHEAD_THRESHOLD)
    {
      #ifdef _OPENMP
      size_t num_threads = omp_get_num_threads();
      size_t tid = omp_get_thread_num();
      #else
      size_t num_threads = 1;
      size_t tid = 0;
      #endif
      ptrdiff_t i = tid * (sz / num_threads);
      ptrdiff_t i_end = tid == num_threads - 1 ? sz : i + sz / num_threads;
      __m256 YMM15 = _mm256_set_ps(value, value, value, value, value, value, value, value);
      __m256 YMM0, YMM1, YMM2, YMM3, YMM4, YMM5;
      for (; i<=((i_end)-16); i+=16) {
        YMM0 = _mm256_loadu_ps(tp+i);
        YMM1 = _mm256_loadu_ps(tp+i+8);
        YMM4 = _mm256_div_ps(YMM0, YMM15);
        YMM5 = _mm256_div_ps(YMM1, YMM15);
        _mm256_storeu_ps(rp+i, YMM4);
        _mm256_storeu_ps(rp+i+8, YMM5);
      }
      for (; i<i_end; i++) {
        rp[i] = tp[i] / value;
      }
    }
  } else {
    TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

#endif

#endif
