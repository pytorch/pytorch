#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>
#if !(defined(__VSX__)  || defined(CPU_CAPABILITY_VSX) || defined(CPU_CAPABILITY_ZVECTOR))
#if defined(CPU_CAPABILITY_SVE256)
#include <ATen/cpu/vec/sve/vec_common_sve.h>
#endif
#include <ATen/cpu/vec/vec256/vec256_float.h>
#include <ATen/cpu/vec/vec256/vec256_bfloat16.h>
#include <ATen/cpu/vec/vec256/vec256_double.h>
#include <ATen/cpu/vec/vec256/vec256_int.h>
#include <ATen/cpu/vec/vec256/vec256_qint.h>
#include <ATen/cpu/vec/vec256/vec256_complex_float.h>
#include <ATen/cpu/vec/vec256/vec256_complex_double.h>
#elif defined(__VSX__)  || defined(CPU_CAPABILITY_VSX)
#include <ATen/cpu/vec/vec256/vsx/vec256_common_vsx.h>
#else
#include <ATen/cpu/vec/vec256/zarch/vec256_zarch.h>
#include <ATen/cpu/vec/vec256/vec256_bfloat16.h>
#endif

#include <ATen/cpu/vec/vec256/vec256_convert.h>
#include <ATen/cpu/vec/vec256/vec256_mask.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ostream>

namespace at::vec {

// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

inline std::ostream& operator<<(std::ostream& stream, const c10::qint32& val) {
  stream << val.val_;
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::qint8& val) {
  stream << static_cast<int>(val.val_);
  return stream;
}
inline std::ostream& operator<<(std::ostream& stream, const c10::quint8& val) {
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


#if defined(CPU_CAPABILITY_AVX2)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX2) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm256_castpd_ps(src);
}

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm256_castps_pd(src);
}

template<>
inline Vectorized<float> cast<float, int32_t>(const Vectorized<int32_t>& src) {
  return _mm256_castsi256_ps(src);
}

template<>
inline Vectorized<double> cast<double, int64_t>(const Vectorized<int64_t>& src) {
  return _mm256_castsi256_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef _MSC_VER
// MSVC is not working well on complex function overload.
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
#endif
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#ifndef _MSC_VER
// MSVC is not working well on complex function overload.
template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex, Vectorized<double>& mask) {
  return _mm256_mask_i64gather_pd(src, base_addr, vindex, mask, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex, Vectorized<float>& mask) {
  return _mm256_mask_i32gather_ps(src, base_addr, vindex, mask, scale);
}
#endif
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

// From: https://stackoverflow.com/a/41148578
template<>
Vectorized<double>
inline convert_to_fp_of_same_size<double>(const Vectorized<int64_t> &src) {
  __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000); /* 2^52 */
  __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000080000000); /* 2^84 + 2^63 */
  __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000080100000); /* 2^84 + 2^63 + 2^52 */
  __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

  __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, src, 0b01010101); /* v_low = low32 + 2^52 */
  __m256i v_hi         = _mm256_srli_epi64(src, 32);
          v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32); /* v_hi = high32*2^32 + 2^84 + 2^63 */
  /* int64 = low32 + high32*2^32 = v_hi + v_lo - 2^52 - 2^63 - 2^84 */
  __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all);
  __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));
  return result;
}

template<>
Vectorized<float>
inline convert_to_fp_of_same_size<float>(const Vectorized<int32_t> &src) {
  return _mm256_cvtepi32_ps(src);
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLIP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> flip(const Vectorized<float> & v) {
  const __m256i mask_float = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_ps(v, mask_float);
}

template<>
inline Vectorized<double> flip(const Vectorized<double> & v) {
  return _mm256_permute4x64_pd(v, 27);  // 27 == _MM_SHUFFLE(0, 1, 2, 3)
}

template<>
inline Vectorized<int64_t> flip(const Vectorized<int64_t> & v) {
  return _mm256_permute4x64_epi64(v, 27);  // 27 == _MM_SHUFFLE(0, 1, 2, 3)
}

template<>
inline Vectorized<int32_t> flip(const Vectorized<int32_t> & v) {
  const __m256i mask_int32 = _mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  return _mm256_permutevar8x32_epi32(v, mask_int32);
}

template<>
inline Vectorized<int16_t> flip(const Vectorized<int16_t> & v) {
  const __m256i mask = _mm256_set_epi8(
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14
  );
  auto reversed = _mm256_shuffle_epi8(v, mask);
  return _mm256_permute2x128_si256(reversed, reversed, 1);
}

inline __m256i flip8(const __m256i & v) {
  const __m256i mask_int8 = _mm256_set_epi8(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  );
  auto reversed = _mm256_shuffle_epi8(v, mask_int8);
  return _mm256_permute2x128_si256(reversed, reversed, 1);
}

template<>
inline Vectorized<int8_t> flip(const Vectorized<int8_t> & v) {
  return flip8(v);
}

template<>
inline Vectorized<uint8_t> flip(const Vectorized<uint8_t> & v) {
  return flip8(v);
}

inline Vectorized<bool> operator&&(
    const Vectorized<bool>& self,
    const Vectorized<bool>& other) {
  const __m256i* self_ = reinterpret_cast<const __m256i*>(self.as_bytes());
  const __m256i* other_ = reinterpret_cast<const __m256i*>(other.as_bytes());
  __m256i out = _mm256_and_si256(*self_, *other_);
  Vectorized<bool> ret;
  std::memcpy(ret, &out, ret.size() * sizeof(bool));
  return ret;
}

#endif // (defined(CPU_CAPABILITY_AVX2)

}} // namepsace at::vec::CPU_CAPABILITY
