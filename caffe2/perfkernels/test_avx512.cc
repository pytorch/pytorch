#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
// check avx512f
__m512 addConstant(__m512 arg) {
  return _mm512_add_ps(arg, _mm512_set1_ps(1.f));
}
// check avx512dq
__m512 andConstant(__m512 arg) {
  return _mm512_and_ps(arg, _mm512_set1_ps(1.f));
}
int main() {
  __m512i a = _mm512_set1_epi32(1);
  __m256i ymm = _mm512_extracti64x4_epi64(a, 0);
  ymm = _mm256_abs_epi64(ymm); // check avx512vl
  __mmask16 m = _mm512_cmp_epi32_mask(a, a, _MM_CMPINT_EQ);
  __m512i r = _mm512_andnot_si512(a, a);
}
