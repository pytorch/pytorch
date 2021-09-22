#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/vec512/vec512_float.h>
#include <ATen/cpu/vec/vec512/vec512_bfloat16.h>
#include <ATen/cpu/vec/vec512/vec512_double.h>
#include <ATen/cpu/vec/vec512/vec512_int.h>
#include <ATen/cpu/vec/vec512/vec512_qint.h>
#include <ATen/cpu/vec/vec512/vec512_complex_float.h>
#include <ATen/cpu/vec/vec512/vec512_complex_double.h>


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace at {
namespace vec {

// See Note [Acceptable use of anonymous namespace in header]
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


#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST (AVX512) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return _mm512_castpd_ps(src);
}

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return _mm512_castps_pd(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex) {
  return _mm512_i64gather_pd(vindex, base_addr, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex) {
  return _mm512_i32gather_ps(vindex, base_addr, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex, const Vectorized<double>& mask) {
  auto all_ones = _mm512_castsi512_pd(_mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
  auto mask_ = _mm512_cmp_pd_mask(all_ones, mask.values, _CMP_EQ_OQ);
  return _mm512_mask_i64gather_pd(src, mask_, vindex, base_addr, scale);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex, const Vectorized<float>& mask) {
  auto all_ones = _mm512_castsi512_ps(_mm512_set1_epi32(0xFFFFFFFF));
  auto mask_ = _mm512_cmp_ps_mask(all_ones, mask.values, _CMP_EQ_OQ);
  return _mm512_mask_i32gather_ps(src, mask_, vindex, base_addr, scale);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
Vectorized<int64_t>
inline convert_to_int_of_same_size<double>(const Vectorized<double> &src) {
  return _mm512_cvtpd_epi64(src);
}

template<>
Vectorized<int32_t>
inline convert_to_int_of_same_size<float>(const Vectorized<float> &src) {
  return _mm512_cvttps_epi32(src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline interleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, a1, a3, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  __m512i idx1 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  __m512i idx2 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  return std::make_pair(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                        _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  //
  //  return:
  //    {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
  //    {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
  __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4,
                                  19, 3, 18, 2, 17, 1, 16, 0);
  __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12,
                                  27, 11, 26, 10, 25, 9, 24, 8);
  return std::make_pair(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                        _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline deinterleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}
  // output:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  // The members of indices have been written in binary format for better understandability
  __m512i idx1 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idx2 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);

  return std::make_pair(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                        _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline deinterleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3, a4, b4, a5, b5, a6, b6, a7, b7}
  //   b = {a8, b8, a9, b9, a10, b10, a11, b11, a12, b12, a13, b13, a14, b14, a15, b15}
  // output:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15}
  //          {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15}
  __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16,
                                  14, 12, 10, 8, 6, 4, 2, 0);
  __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17,
                                  15, 13, 11, 9, 7, 5, 3, 1);

  return std::make_pair(_mm512_mask_permutex2var_ps(a, 0xffff, idx1, b),
                        _mm512_mask_permutex2var_ps(a, 0xffff, idx2, b));
}

#endif // defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

}}}
