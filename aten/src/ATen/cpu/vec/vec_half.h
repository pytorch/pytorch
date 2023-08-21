#pragma once

#include <ATen/cpu/vec/intrinsics.h>

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
static inline uint16_t float2half_scalar(float val) {
#if defined(CPU_CAPABILITY_AVX2)
#if defined(_MSC_VER)
  __m256 v = _mm256_set1_ps(val);
  __m128i o =
      _mm256_cvtps_ph(v, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return static_cast<std::uint16_t>(_mm_cvtsi128_si32(o));
#else
  return _cvtss_sh(val, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
#endif
#elif defined(CPU_CAPABILITY_AVX512)
  __m512 v = _mm512_set1_ps(val);
  __m256i o =
      _mm512_cvtps_ph(v, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  return static_cast<std::uint16_t>(
      _mm_cvtsi128_si32(_mm256_castsi256_si128(o)));
#endif
}

static inline float half2float_scalar(uint16_t val) {
#if defined(CPU_CAPABILITY_AVX2)
#if defined(_MSC_VER)
  __m128i v = _mm_cvtsi32_si128(val);
  __m256 o = _mm256_cvtph_ps(v);
  return _mm256_cvtss_f32(o);
#else
  return _cvtsh_ss(val);
#endif
#elif defined(CPU_CAPABILITY_AVX512)
  __m256i v =
      _mm256_setr_epi16(val, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m512 o = _mm512_cvtph_ps(v);
  return _mm512_cvtss_f32(o);
#endif
}

#endif

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
