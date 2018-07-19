#include "caffe2/core/types.h"
#include "caffe2/perfkernels/cvtsh_ss_bugfix.h"
#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/utils/math.h"

#include <emmintrin.h>
#include <immintrin.h>

namespace caffe2 {

void TypedAxpy_float16_float__avx2_fma(
    int N,
    const float a,
    const float16* x,
    float* y) {
  // if x does not start at the 16 byte boundary, we will process the first few.
  // before we get to a real one.
  while (((unsigned long)x % 16) && N) {
    *(y++) += _cvtsh_ss((*(x++)).x) * a;
    --N;
  }

  // From now on we can do vectorized additions using __m256, which is 8 floats,
  // so we will vectorize every 8 element and then resort to cvtsh_ss.
  __m256 mma = _mm256_set1_ps(a);
  int current = 0;
  const int bound = (N % 8) ? N - 8 : N;

  for (; current < bound; current += 8) {
    __m128i mmx_16 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + current));
    __m256 mmx_32 = _mm256_cvtph_ps(mmx_16);
    __m256 mmy = _mm256_loadu_ps(y + current);
    mmy = _mm256_fmadd_ps(mmx_32, mma, mmy);
    _mm256_storeu_ps(y + current, mmy);
  }

  if (bound != N) {
    while (current < N) {
      y[current] += _cvtsh_ss(x[current].x) * a;
      ++current;
    }
  }
}

void TypedAxpy_uint8_float__avx2_fma(
    int N,
    const float a,
    const std::uint8_t* x,
    float* y) {
  // if x does not start at the 16 byte boundary, we will process the first few.
  // before we get to a real one.
  while (((unsigned long)x % 16) && N) {
    *(y++) += (float)(*(x++)) * a;
    --N;
  }

  // From now on we can do vectorized additions using __m256, which is 8 floats,
  // so we will vectorize every 8 element and then resort to cvtsh_ss.
  __m256 mma = _mm256_set1_ps(a);
  int current = 0;
  const int bound = (N % 8) ? N - 8 : N;

  for (; current < bound; current += 8) {
    __m256i mmx_int32 = _mm256_cvtepi8_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + current)));
    __m256 mmx_fp32 = _mm256_cvtepi32_ps(mmx_int32);

    __m256 mmy = _mm256_loadu_ps(y + current);
    mmy = _mm256_fmadd_ps(mmx_fp32, mma, mmy);
    _mm256_storeu_ps(y + current, mmy);
  }

  if (bound != N) {
    while (current < N) {
      y[current] += (float)(x[current]) * a;
      ++current;
    }
  }
}

} // namespace caffe2
