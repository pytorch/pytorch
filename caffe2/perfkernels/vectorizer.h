#pragma once

#if (ENABLE_VECTORIZATION > 0) && !defined(_DEBUG) && !defined(DEBUG)
#if defined(__clang__) && (__clang_major__ > 7)
#define IS_SANITIZER                          \
  ((__has_feature(address_sanitizer) == 1) || \
   (__has_feature(memory_sanitizer) == 1) ||  \
   (__has_feature(thread_sanitizer) == 1) ||  \
   (__has_feature(undefined_sanitizer) == 1))

#if IS_SANITIZER == 0
#define VECTOR_LOOP _Pragma("clang loop vectorize(enable)")
#define FAST_MATH _Pragma("clang fp contract(fast)")
#define VECTORIZED_KERNEL 1
#endif
#elif defined(_OPENMP) && (_OPENMP >= 201511)
// Support with OpenMP4.5 and above
#define VECTOR_LOOP _Pragma("omp for simd")
#define VECTORIZED_KERNEL 1
#define FAST_MATH
#endif
#endif

#ifndef VECTOR_LOOP
// Not supported
#define VECTOR_LOOP
#define FAST_MATH
#endif
