#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/vec256/intrinsics.h>

#include <ATen/cpu/vec/vec256/vec256_base.h>
#if !defined(__VSX__)  || !defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec/vec256/vec256_float.h>
#include <ATen/cpu/vec/vec256/vec256_float_neon.h>
#include <ATen/cpu/vec/vec256/vec256_bfloat16.h>
#include <ATen/cpu/vec/vec256/vec256_double.h>
#include <ATen/cpu/vec/vec256/vec256_int.h>
#include <ATen/cpu/vec/vec256/vec256_qint.h>
#include <ATen/cpu/vec/vec256/vec256_complex_float.h>
#include <ATen/cpu/vec/vec256/vec256_complex_double.h>
#else
#include <ATen/cpu/vec/vec256/vsx/vec256_common_vsx.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace at {
namespace vec {

// Note [Acceptable use of anonymous namespace in header]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Yes you saw right, this is an anonymous namespace in a header.  This header,
// and all of its subheaders, REQUIRE their code to be entirely inlined into
// the compilation unit that uses them.  It's important that these functions have
// internal linkage so that kernels for different architectures don't get
// combined during linking. It's sufficient to label functions "static", but
// class methods must be an unnamed namespace to have internal linkage (since
// static means something different in the context of classes).
namespace {

 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
     stream << val.val_;
     return stream;
 }
 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
     stream << static_cast<int>(val.val_);
     return stream;
 }
 C10_UNUSED std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
     stream << static_cast<unsigned int>(val.val_);
     return stream;
 }

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Vectorized<T>& vec) {
  T buf[Vectorized<T>::size()];
  vec.store(buf);
  stream << "vec[";
  for (int i = 0; i != Vectorized<T>::size(); i++) {
    if (i != 0) {
      stream << ", ";
    }
    stream << buf[i];
  }
  stream << "]";
  return stream;
}


#if (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm256_castpd_ps(src);
}

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm256_castps_pd(src);
}

#if defined(CPU_CAPABILITY_AVX2)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define DEFINE_FLOAT_INT_CAST(int_t, float_t, float_ch)            \
template<>                                                         \
inline  Vectorized<int_t> cast<int_t, float_t>(const Vectorized<float_t>& src) {   \
  return _mm256_castp ## float_ch ## _si256(src);                  \
}                                                                  \
template<>                                                         \
inline Vectorized<float_t> cast<float_t, int_t>(const Vectorized<int_t>& src) {   \
  return _mm256_castsi256_p ## float_ch (src);                     \
}

DEFINE_FLOAT_INT_CAST(int64_t, double, d)
DEFINE_FLOAT_INT_CAST(int32_t, double, d)
DEFINE_FLOAT_INT_CAST(int16_t, double, d)
DEFINE_FLOAT_INT_CAST(int64_t, float, s)
DEFINE_FLOAT_INT_CAST(int32_t, float, s)
DEFINE_FLOAT_INT_CAST(int16_t, float, s)

#undef DEFINE_FLOAT_INT_CAST

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm256_i64gather_pd(base_addr, vindex, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm256_i32gather_ps(base_addr, vindex, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex, const Vectorized<double>& mask) {
  return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex, const Vectorized<float>& mask) {
  return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Only works for inputs in the range: [-2^51, 2^51]
// From: https://stackoverflow.com/a/41148578
template<>
Vectorized<int64_t>
inline convert_to_int_of_same_size<double>(const Vectorized<double> &src) {
  auto x = _mm256_add_pd(src, _mm256_set1_pd(0x0018000000000000));
  return _mm256_sub_epi64(
      _mm256_castpd_si256(x),
      _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
  );
}

template<>
Vectorized<int32_t>
inline convert_to_int_of_same_size<float>(const Vectorized<float> &src) {
  return _mm256_cvttps_epi32(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline interleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, a1, a3, a3}
  //   b = {b0, b1, b2, b3}

  // swap lanes:
  //   a_swapped = {a0, a1, b0, b1}
  //   b_swapped = {a2, a3, b2, b3}
  auto a_swapped = _mm256_permute2f128_pd(a, b, 0b0100000);  // 0, 2.   4 bits apart
  auto b_swapped = _mm256_permute2f128_pd(a, b, 0b0110001);  // 1, 3.   4 bits apart

  // group cols crossing lanes:
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(_mm256_permute4x64_pd(a_swapped, 0b11011000),  // 0, 2, 1, 3
                        _mm256_permute4x64_pd(b_swapped, 0b11011000)); // 0, 2, 1, 3
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}

  // swap lanes:
  //   a_swapped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_swapped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: can we support caching this?
  auto a_swapped = _mm256_permute2f128_ps(a, b, 0b0100000);  // 0, 2.   4 bits apart
  auto b_swapped = _mm256_permute2f128_ps(a, b, 0b0110001);  // 1, 3.   4 bits apart

  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  const __m256i group_ctrl = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  return std::make_pair(_mm256_permutevar8x32_ps(a_swapped, group_ctrl),
                        _mm256_permutevar8x32_ps(b_swapped, group_ctrl));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline deinterleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}

  // group cols crossing lanes:
  //   a_grouped = {a0, a1, b0, b1}
  //   b_grouped = {a2, a3, b2, b3}
  auto a_grouped = _mm256_permute4x64_pd(a, 0b11011000);  // 0, 2, 1, 3
  auto b_grouped = _mm256_permute4x64_pd(b, 0b11011000);  // 0, 2, 1, 3

  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(_mm256_permute2f128_pd(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                        _mm256_permute2f128_pd(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline deinterleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}

  // group cols crossing lanes:
  //   a_grouped = {a0, a1, a2, a3, b0, b1, b2, b3}
  //   b_grouped = {a4, a5, a6, a7, b4, b5, b6, b7}
  // TODO: can we support caching this?
  const __m256i group_ctrl = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
  auto a_grouped = _mm256_permutevar8x32_ps(a, group_ctrl);
  auto b_grouped = _mm256_permutevar8x32_ps(b, group_ctrl);

  // swap lanes:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  return std::make_pair(_mm256_permute2f128_ps(a_grouped, b_grouped, 0b0100000),  // 0, 2.   4 bits apart
                        _mm256_permute2f128_ps(a_grouped, b_grouped, 0b0110001)); // 1, 3.   4 bits apart
}

template <>
inline void transpose_kernel_8x8<float>(const float* src, int64_t ld_src, float* dst, int64_t ld_dst) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  //   c = {c0, c1, c2, c3, c4, c5, c6, c7}
  //   d = {d0, d1, d2, d3, d4, d5, d6, d7}
  //   e = {e0, e1, e2, e3, e4, e5, e6, e7}
  //   f = {f0, f1, f2, f3, f4, f5, f6, f7}
  //   g = {g0, g1, g2, g3, g4, g5, g6, g7}
  //   h = {h0, h1, h2, h3, h4, h5, h6, h7}
  __m256 a = _mm256_loadu_ps(&src[0 * ld_src]);
  __m256 b = _mm256_loadu_ps(&src[1 * ld_src]);
  __m256 c = _mm256_loadu_ps(&src[2 * ld_src]);
  __m256 d = _mm256_loadu_ps(&src[3 * ld_src]);
  __m256 e = _mm256_loadu_ps(&src[4 * ld_src]);
  __m256 f = _mm256_loadu_ps(&src[5 * ld_src]);
  __m256 g = _mm256_loadu_ps(&src[6 * ld_src]);
  __m256 h = _mm256_loadu_ps(&src[7 * ld_src]);

  // interleave 32 bit:
  //   t0 = {a0, b0, a1, b1, a4, b4, a5, b5}
  //   t1 = {a2, b2, a3, b3, a6, b6, a7, b7}
  //   t2 = {c0, d0, c1, d1, c4, d4, c5, d5}
  //   t3 = {c2, d2, c3, d3, c6, d6, c7, d7}
  //   t4 = {e0, f0, e1, f1, e4, f4, e5, f5}
  //   t5 = {e2, f2, e3, f3, e6, f6, e7, f7}
  //   t6 = {g0, h0, g1, h1, g4, h4, g5, h5}
  //   t7 = {g2, h2, g3, h3, g6, h6, g7, h7}
  __m256 t0 = _mm256_unpacklo_ps(a, b);
  __m256 t1 = _mm256_unpackhi_ps(a, b);
  __m256 t2 = _mm256_unpacklo_ps(c, d);
  __m256 t3 = _mm256_unpackhi_ps(c, d);
  __m256 t4 = _mm256_unpacklo_ps(e, f);
  __m256 t5 = _mm256_unpackhi_ps(e, f);
  __m256 t6 = _mm256_unpacklo_ps(g, h);
  __m256 t7 = _mm256_unpackhi_ps(g, h);

  // shuffle 64 bit:
  //   tt0 = {a0, b0, c0, d0, a4, b4, c4, d4}
  //   tt1 = {a1, b1, c1, d1, a5, b5, c5, d5}
  //   tt2 = {e0, f0, g0, h0, e4, f4, g4, h4}
  //   tt3 = {e1, f1, g1, h1, e5, b5, c5, d5}
  //   tt4 = {a2, b2, c2, d2, a6, b6, c6, d6}
  //   tt5 = {a3, b3, c3, d3, a7, b7, c7, d7}
  //   tt6 = {e2, f2, g2, h2, e6, f6, g6, h6}
  //   tt7 = {e3, f3, g3, h3, e7, f7, g7, h7}
  __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
  __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xee);
  __m256 tt2 = _mm256_shuffle_ps(t4, t6, 0x44);
  __m256 tt3 = _mm256_shuffle_ps(t4, t6, 0xee);
  __m256 tt4 = _mm256_shuffle_ps(t1, t3, 0x44);
  __m256 tt5 = _mm256_shuffle_ps(t1, t3, 0xee);
  __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
  __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xee);

  // swap 128 bit:
  //   a = {a0, b0, c0, d0, e0, f0, g0, h0}
  //   b = {a1, b1, c1, d1, e1, f1, g1, h1}
  //   c = {a2, b2, c2, d2, e2, f2, g2, h2}
  //   d = {a3, b3, c3, d3, e3, f3, g3, h3}
  //   e = {a4, b4, c4, d4, e4, f4, g4, h4}
  //   f = {a5, b5, c5, d5, e5, f5, g5, h5}
  //   g = {a6, b6, c6, d6, e6, f6, g6, h6}
  //   h = {a7, b7, c7, d7, e7, f7, g7, h7}
  a = _mm256_permute2f128_ps(tt0, tt2, 0x20);
  b = _mm256_permute2f128_ps(tt1, tt3, 0x20);
  c = _mm256_permute2f128_ps(tt4, tt6, 0x20);
  d = _mm256_permute2f128_ps(tt5, tt7, 0x20);
  e = _mm256_permute2f128_ps(tt0, tt2, 0x31);
  f = _mm256_permute2f128_ps(tt1, tt3, 0x31);
  g = _mm256_permute2f128_ps(tt4, tt6, 0x31);
  h = _mm256_permute2f128_ps(tt5, tt7, 0x31);

  _mm256_storeu_ps(&dst[0 * ld_dst], a);
  _mm256_storeu_ps(&dst[1 * ld_dst], b);
  _mm256_storeu_ps(&dst[2 * ld_dst], c);
  _mm256_storeu_ps(&dst[3 * ld_dst], d);
  _mm256_storeu_ps(&dst[4 * ld_dst], e);
  _mm256_storeu_ps(&dst[5 * ld_dst], f);
  _mm256_storeu_ps(&dst[6 * ld_dst], g);
  _mm256_storeu_ps(&dst[7 * ld_dst], h);
}

template <>
inline void transpose_kernel_8x8<BFloat16>(const BFloat16* src, int64_t ld_src, BFloat16* dst, int64_t ld_dst) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  //   c = {c0, c1, c2, c3, c4, c5, c6, c7}
  //   d = {d0, d1, d2, d3, d4, d5, d6, d7}
  //   e = {e0, e1, e2, e3, e4, e5, e6, e7}
  //   f = {f0, f1, f2, f3, f4, f5, f6, f7}
  //   g = {g0, g1, g2, g3, g4, g5, g6, g7}
  //   h = {h0, h1, h2, h3, h4, h5, h6, h7}
  __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[0 * ld_src]));
  __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[1 * ld_src]));
  __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[2 * ld_src]));
  __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[3 * ld_src]));
  __m128i e = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[4 * ld_src]));
  __m128i f = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[5 * ld_src]));
  __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[6 * ld_src]));
  __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&src[7 * ld_src]));

  // interleave 16 bit:
  //   t0 = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   t1 = {a4, b4, a5, b5, a6, b6, a7, b7}
  //   t2 = {c0, d0, c1, d1, c2, d2, c3, d3}
  //   t3 = {c4, d4, c5, d5, c6, d6, c7, d7}
  //   t4 = {e0, f0, e1, f1, e2, f2, e3, f3}
  //   t5 = {e4, f4, e5, f5, e6, f6, e7, f7}
  //   t6 = {g0, h0, g1, h1, g2, h2, g3, h3}
  //   t7 = {g4, h4, g5, h5, g6, h6, g7, h7}
  __m128i t0 = _mm_unpacklo_epi16(a, b);
  __m128i t1 = _mm_unpackhi_epi16(a, b);
  __m128i t2 = _mm_unpacklo_epi16(c, d);
  __m128i t3 = _mm_unpackhi_epi16(c, d);
  __m128i t4 = _mm_unpacklo_epi16(e, f);
  __m128i t5 = _mm_unpackhi_epi16(e, f);
  __m128i t6 = _mm_unpacklo_epi16(g, h);
  __m128i t7 = _mm_unpackhi_epi16(g, h);

  // interleave 32 bit:
  //   tt0 = {a0, b0, c0, d0, a1, b1, c1, d1}
  //   tt1 = {a2, b2, c2, d2, a3, b3, c3, d3}
  //   tt2 = {a4, b4, c4, d4, a5, b5, c5, d5}
  //   tt3 = {a6, b6, c6, d6, a7, b7, c7, d7}
  //   tt4 = {e0, f0, g0, h0, e1, f1, g1, g1}
  //   tt5 = {e2, f2, g2, h2, e3, f3, g3, h3}
  //   tt6 = {e4, f4, g4, h4, e5, f5, g5, h5}
  //   tt7 = {e6, f6, g6, h6, e7, f7, g7, h7}
  __m128i tt0 = _mm_unpacklo_epi32(t0, t2);
  __m128i tt1 = _mm_unpackhi_epi32(t0, t2);
  __m128i tt2 = _mm_unpacklo_epi32(t1, t3);
  __m128i tt3 = _mm_unpackhi_epi32(t1, t3);
  __m128i tt4 = _mm_unpacklo_epi32(t4, t6);
  __m128i tt5 = _mm_unpackhi_epi32(t4, t6);
  __m128i tt6 = _mm_unpacklo_epi32(t5, t7);
  __m128i tt7 = _mm_unpackhi_epi32(t5, t7);

  // interleave 64 bit:
  //   a = {a0, b0, c0, d0, e0, f0, g0, h0}
  //   b = {a1, b1, c1, d1, e1, f1, g1, h1}
  //   c = {a2, b2, c2, d2, e2, f2, g2, h2}
  //   d = {a3, b3, c3, d3, e3, f3, g3, h3}
  //   e = {a4, b4, c4, d4, e4, f4, g4, h4}
  //   f = {a5, b5, c5, d5, e5, f5, g5, h5}
  //   g = {a6, b6, c6, d6, e6, f6, g6, h6}
  //   h = {a7, b7, c7, d7, e7, f7, g7, h7}
  a = _mm_unpacklo_epi64(tt0, tt4);
  b = _mm_unpackhi_epi64(tt0, tt4);
  c = _mm_unpacklo_epi64(tt1, tt5);
  d = _mm_unpackhi_epi64(tt1, tt5);
  e = _mm_unpacklo_epi64(tt2, tt6);
  f = _mm_unpackhi_epi64(tt2, tt6);
  g = _mm_unpacklo_epi64(tt3, tt7);
  h = _mm_unpackhi_epi64(tt3, tt7);

  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[0 * ld_dst]), a);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[1 * ld_dst]), b);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[2 * ld_dst]), c);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[3 * ld_dst]), d);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[4 * ld_dst]), e);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[5 * ld_dst]), f);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[6 * ld_dst]), g);
  _mm_storeu_si128(reinterpret_cast<__m128i*>(&dst[7 * ld_dst]), h);
}

#endif  // defined(CPU_CAPABILITY_AVX2)

#endif // (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

}}}
