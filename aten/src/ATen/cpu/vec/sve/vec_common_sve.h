#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with SVE]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/sve/sve_helper.h>

#include <ATen/cpu/vec/sve/vec_float.h>
#include <ATen/cpu/vec/sve/vec_double.h>
#include <ATen/cpu/vec/sve/vec_int.h>
#include <ATen/cpu/vec/sve/vec_qint.h>
#include <ATen/cpu/vec/sve/vec_bfloat16.h>

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
  stream << ']';
  return stream;
}

#if defined(CPU_CAPABILITY_SVE)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<>
inline Vectorized<float> cast<float, double>(const Vectorized<double>& src) {
  return svreinterpret_f32_f64(src);
}

template<>
inline Vectorized<double> cast<double, float>(const Vectorized<float>& src) {
  return svreinterpret_f64_f32(src);
}

#define DEFINE_FLOAT_INT_CAST(int_t, int_bit, float_t, float_bit)                \
template<>                                                                       \
inline  Vectorized<int_t> cast<int_t, float_t>(const Vectorized<float_t>& src) { \
  return svreinterpret_s##int_bit##_f##float_bit(src);                           \
}                                                                                \
template<>                                                                       \
inline Vectorized<float_t> cast<float_t, int_t>(const Vectorized<int_t>& src) {  \
  return svreinterpret_f##float_bit##_s##int_bit(src);                           \
}

DEFINE_FLOAT_INT_CAST(int64_t, 64, double, 64)
DEFINE_FLOAT_INT_CAST(int32_t, 32, double, 64)
DEFINE_FLOAT_INT_CAST(int16_t, 16, double, 64)
DEFINE_FLOAT_INT_CAST(int64_t, 64, float, 32)
DEFINE_FLOAT_INT_CAST(int32_t, 32, float, 32)
DEFINE_FLOAT_INT_CAST(int16_t, 16, float, 32)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline gather(const double* base_addr, const Vectorized<int64_t>& vindex_) {
    svint64_t offsets = svmul_s64_x(ptrue, vindex_, svdup_n_s64(scale));
    return svld1_gather_s64offset_f64(ptrue, base_addr, offsets);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline gather(const float* base_addr, const Vectorized<int32_t>& vindex_) {
    svint32_t offsets = svmul_s32_x(ptrue, vindex_, svdup_n_s32(scale));
    return svld1_gather_s32offset_f32(ptrue, base_addr, offsets);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>>
inline mask_gather(const Vectorized<double>& src, const double* base_addr,
                   const Vectorized<int64_t>& vindex_, const Vectorized<double>& mask_) {
    svbool_t valid_mask = svcmpeq_s64(ptrue, svreinterpret_s64_f64(mask_), ALL_S64_TRUE_MASK);
    svint64_t offsets = svmul_s64_x(ptrue, vindex_, svdup_n_s64(scale));
    svfloat64_t gathered = svld1_gather_s64offset_f64(valid_mask, base_addr, offsets);
    return svsel_f64(valid_mask, gathered, src);
}

template<int64_t scale = 1>
std::enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>>
inline mask_gather(const Vectorized<float>& src, const float* base_addr,
                   const Vectorized<int32_t>& vindex_, const Vectorized<float>& mask_) {
    svbool_t valid_mask = svcmpeq_s32(ptrue, svreinterpret_s32_f32(mask_), ALL_S32_TRUE_MASK);
    svint32_t offsets = svmul_s32_x(ptrue, vindex_, svdup_n_s32(scale));
    svfloat32_t gathered = svld1_gather_s32offset_f32(valid_mask, base_addr, offsets);
    return svsel_f32(valid_mask, gathered, src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Only works for inputs in the range: [-2^51, 2^51]
// From: https://stackoverflow.com/a/41148578
template<>
Vectorized<int64_t>
inline convert_to_int_of_same_size<double>(const Vectorized<double> &src) {
  svfloat64_t x = svadd_f64_x(ptrue, src, svdup_n_f64(0x0018000000000000));
  return svsub_s64_x(ptrue,
                     svreinterpret_s64_f64(x),
                     svreinterpret_s64_f64(svdup_n_f64(0x0018000000000000)));
}

template<>
Vectorized<int32_t>
inline convert_to_int_of_same_size<float>(const Vectorized<float> &src) {
  return svcvt_s32_f32_x(ptrue, src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline interleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, a1, a3, a3}
  //   b = {b0, b1, b2, b3}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(Vectorized<double>(svzip1_f64(a, b)),
                        Vectorized<double>(svzip2_f64(a, b)));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline interleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  return std::make_pair(Vectorized<float>(svzip1_f32(a, b)),
                        Vectorized<float>(svzip2_f32(a, b)));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>>
inline deinterleave2<double>(const Vectorized<double>& a, const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(Vectorized<double>(svuzp1_f64(a, b)),
                        Vectorized<double>(svuzp2_f64(a, b)));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>>
inline deinterleave2<float>(const Vectorized<float>& a, const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}
  // swap lanes:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  return std::make_pair(Vectorized<float>(svuzp1_f32(a, b)),
                        Vectorized<float>(svuzp2_f32(a, b)));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLIP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define DEFINE_FLIP_FUNC(type, sve_func)        \
inline Vectorized<type> flip(const Vectorized<type> & v) {  \
    return Vectorized<type>(sve_func(v));      \
}
// Use the macro to define the flip functions
DEFINE_FLIP_FUNC(float, svrev_f32)
DEFINE_FLIP_FUNC(double, svrev_f64)
DEFINE_FLIP_FUNC(int64_t, svrev_s64)
DEFINE_FLIP_FUNC(int32_t, svrev_s32)
DEFINE_FLIP_FUNC(int16_t, svrev_s16)
DEFINE_FLIP_FUNC(int8_t, svrev_s8)

#endif // defined(CPU_CAPABILITY_SVE)

}}
