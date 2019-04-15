#include "transpose.h"

#include <x86intrin.h>

namespace fbgemm {

void transpose_4rows(int N, const std::uint8_t* src, std::uint8_t* dst) {
  constexpr int M = 4;
  int j;
  // vectorized loop
  for (j = 0; j < N / 32 * 32; j += 32) {
    // a : a0 a1 ... a31
    // b : b0 b1 ... b31
    // c : c0 c1 ... c31
    // d : d0 d1 ... d31
    __m256i a = _mm256_lddqu_si256((const __m256i*)(src + j + 0 * N));
    __m256i b = _mm256_lddqu_si256((const __m256i*)(src + j + 1 * N));
    __m256i c = _mm256_lddqu_si256((const __m256i*)(src + j + 2 * N));
    __m256i d = _mm256_lddqu_si256((const __m256i*)(src + j + 3 * N));

    // even-odd interleaving
    // ab_lo : a0 b0 a1 b1 ...  a7  b7 | a16 b16 ... a23 b23
    // ab_hi : a8 b8 a9 b9 ... a15 b15 | a24 b24 ... a31 b31
    // cd_lo : c0 d0 c1 d1 ...  c7  d7 | c16 d16 ... c23 d23
    // cd_hi : c8 d8 c9 d9 ... c15 d15 | c24 d24 ... c31 d31
    __m256i ab_lo = _mm256_unpacklo_epi8(a, b);
    __m256i ab_hi = _mm256_unpackhi_epi8(a, b);
    __m256i cd_lo = _mm256_unpacklo_epi8(c, d);
    __m256i cd_hi = _mm256_unpackhi_epi8(c, d);

    // 4-row interleaving but permuted at 128-bit granularity
    // y0 :  a0  b0  c0  d0 ...  a-d3 | a-d16 ... a-d19
    // y1 :  a4  b4  c4  d4 ...  a-d7 | a-d20 ... a-d23
    // y2 :  a8  b8  c8  d8 ... a-d11 | a-d24 ... a-d27
    // y3 : a12 b12 c12 d12 ... a-d15 | a-d28 ... a-d31
    __m256i y0 = _mm256_unpacklo_epi16(ab_lo, cd_lo);
    __m256i y1 = _mm256_unpackhi_epi16(ab_lo, cd_lo);
    __m256i y2 = _mm256_unpacklo_epi16(ab_hi, cd_hi);
    __m256i y3 = _mm256_unpackhi_epi16(ab_hi, cd_hi);

    // Storing with 128-bit lanes are permuted so that everything is in order
    _mm256_storeu_si256(
        (__m256i*)(dst + j * M + 0 * 32),
        _mm256_permute2f128_si256(y0, y1, 0x20));
    _mm256_storeu_si256(
        (__m256i*)(dst + j * M + 1 * 32),
        _mm256_permute2f128_si256(y2, y3, 0x20));
    _mm256_storeu_si256(
        (__m256i*)(dst + j * M + 2 * 32),
        _mm256_permute2f128_si256(y0, y1, 0x31));
    _mm256_storeu_si256(
        (__m256i*)(dst + j * M + 3 * 32),
        _mm256_permute2f128_si256(y2, y3, 0x31));
  }
  // scalar loop for remainder
  for (; j < N; ++j) {
    for (int i = 0; i < M; ++i) {
      dst[j * M + i] = src[j + i * N];
    }
  }
}

} // namespace fbgemm
