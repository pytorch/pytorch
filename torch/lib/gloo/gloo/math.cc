#include "gloo/math.h"

#include <algorithm>

#ifdef GLOO_USE_AVX
#include <immintrin.h>
#endif

#include "gloo/types.h"

#define is_aligned(POINTER, BYTE_COUNT) \
  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

namespace gloo {

#ifdef GLOO_USE_AVX

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void sum<float16>(float16* x, const float16* y, size_t n) {
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_add_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] += y[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void product<float16>(float16* x, const float16* y, size_t n) {
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_mul_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] *= y[i];
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void max<float16>(float16* x, const float16* y, size_t n) {
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_max_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] = std::max(x[i], y[i]);
  }
}

// Assumes x and y are either both aligned to 32 bytes or unaligned by the same
// offset, as would happen when reducing at an offset within an aligned buffer
template <>
void min<float16>(float16* x, const float16* y, size_t n) {
  size_t i;
  for (i = 0; i < (n / 8) * 8; i += 8) {
    __m256 va32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&x[i])));
    __m256 vb32 = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(&y[i])));
    __m128i vc16 = _mm256_cvtps_ph(_mm256_min_ps(va32, vb32), 0);
    _mm_storeu_si128((__m128i*)(&x[i]), vc16);
  }
  // Leftovers
  for (; i < n; i++) {
    x[i] = std::min(x[i], y[i]);
  }
}

#endif

}
