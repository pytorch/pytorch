// Common utilities for writing performance kernels and easy dispatching of
// different backends.
/*
The general workflow shall be as follows, say we want to
implement a functionality called void foo(int a, float b).

In foo.h, do:
   void foo(int a, float b);

In foo_avx2.cc, do:
   void foo__avx2(int a, float b) {
     [actual avx2 implementation]
   }

In foo_avx.cc, do:
   void foo__avx(int a, float b) {
     [actual avx implementation]
   }

In foo.cc, do:
   // The base implementation should *always* be provided.
   void foo__base(int a, float b) {
     [base, possibly slow implementation]
   }
   void foo(int a, float b) {
     // You should always order things by their preference, faster
     // implementations earlier in the function.
     AVX2_DO(foo, a, b);
     AVX_DO(foo, a, b);
     BASE_DO(foo, a, b);
   }

*/
// Details: this functionality basically covers the cases for both build time
// and run time architecture support.
//
// During build time:
//    The build system should provide flags CAFFE2_PERF_WITH_AVX2 and
//    CAFFE2_PERF_WITH_AVX that corresponds to the __AVX__ and __AVX2__ flags
//    the compiler provides. Note that we do not use the compiler flags but
//    rely on the build system flags, because the common files (like foo.cc
//    above) will always be built without __AVX__ and __AVX2__.
// During run time:
//    we use cpuid to identify cpu support and run the proper functions.

#pragma once

// DO macros: these should be used in your entry function, similar to foo()
// above, that routes implementations based on CPU capability.

#define BASE_DO(funcname, ...) return funcname##__base(__VA_ARGS__);

#ifdef CAFFE2_PERF_WITH_AVX2
#define AVX2_DO(funcname, ...)                 \
  decltype(funcname##__base) funcname##__avx2; \
  if (GetCpuId().avx2()) {                     \
    return funcname##__avx2(__VA_ARGS__);      \
  }
#define AVX2_FMA_DO(funcname, ...)                 \
  decltype(funcname##__base) funcname##__avx2_fma; \
  if (GetCpuId().avx2() && GetCpuId().fma()) {     \
    return funcname##__avx2_fma(__VA_ARGS__);      \
  }
#else // CAFFE2_PERF_WITH_AVX2
#define AVX2_DO(funcname, ...)
#define AVX2_FMA_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX2

#ifdef CAFFE2_PERF_WITH_AVX
#define AVX_DO(funcname, ...)                 \
  decltype(funcname##__base) funcname##__avx; \
  if (GetCpuId().avx()) {                     \
    return funcname##__avx(__VA_ARGS__);      \
  }
#define AVX_F16C_DO(funcname, ...)                 \
  decltype(funcname##__base) funcname##__avx_f16c; \
  if (GetCpuId().avx() && GetCpuId().f16c()) {     \
    return funcname##__avx_f16c(__VA_ARGS__);      \
  }
#else // CAFFE2_PERF_WITH_AVX
#define AVX_DO(funcname, ...)
#define AVX_F16C_DO(funcname, ...)
#endif // CAFFE2_PERF_WITH_AVX
