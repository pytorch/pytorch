#pragma once // TODO: remove this entire file


#define INT16 short
#define INT32 int

#define UINT16 unsigned INT16
#define UINT32 unsigned INT32

/* Microsoft compiler doesn't limit intrinsics for an architecture.
   This macro is set only on x86 and means SSE2 and above including AVX2. */
#if defined(_M_X64) || _M_IX86_FP == 2
    #define __SSE4_2__
#endif

#if defined(__SSE4_2__)
    #include <emmintrin.h>
    #include <mmintrin.h>
    #include <smmintrin.h>
#endif

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

#if defined(__SSE4_2__)
static __m128i inline
mm_cvtepu8_epi32(void *ptr) {
    return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(INT32 *) ptr));
}
#endif

#if defined(__AVX2__)
static __m256i inline
mm256_cvtepu8_epi32(void *ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i *) ptr));
}
#endif
