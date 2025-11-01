#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with SVE]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>

#ifdef CPU_CAPABILITY_SVE128

#include <ATen/cpu/vec/vec128/vec128_float_neon.h>

#include <ATen/cpu/vec/vec128/vec128_bfloat16_neon.h>

#include <ATen/cpu/vec/vec128/vec128_half_neon.h>

#include <ATen/cpu/vec/vec128/vec128_convert.h>

#include <ATen/cpu/vec/sve/vec_qint.h>

#elif defined(CPU_CAPABILITY_SVE)

#include <ATen/cpu/vec/sve/vec_float.h>

#include <ATen/cpu/vec/sve/vec_bfloat16.h>

#include <ATen/cpu/vec/sve/vec_double.h>
#include <ATen/cpu/vec/sve/vec_int.h>

#include <ATen/cpu/vec/sve/vec_qint.h>

#include <ATen/cpu/vec/vec256/vec256_half.h>

#include <ATen/cpu/vec/vec256/vec256_convert.h>

#else // NEON

#include <ATen/cpu/vec/vec128/vec128_float_neon.h>

#include <ATen/cpu/vec/vec128/vec128_half_neon.h>

#include <ATen/cpu/vec/vec128/vec128_bfloat16_neon.h>

#include <ATen/cpu/vec/vec128/vec128_convert.h>

#include <ATen/cpu/vec/vec256/vec256_qint.h>

#endif // defined(CPU_CAPABILITY_SVE128)

#include <ATen/cpu/vec/functional.h>

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

#if defined(CPU_CAPABILITY_SVE)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CAST ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define DEFINE_SVE_CAST(t1_t, t1_prefix, t2_t, t2_prefix)                 \
  template <>                                                             \
  inline Vectorized<t1_t> cast<t1_t, t2_t>(const Vectorized<t2_t>& src) { \
    return svreinterpret_##t1_prefix##_##t2_prefix(src);                  \
  }                                                                       \
  template <>                                                             \
  inline Vectorized<t2_t> cast<t2_t, t1_t>(const Vectorized<t1_t>& src) { \
    return svreinterpret_##t2_prefix##_##t1_prefix(src);                  \
  }

DEFINE_SVE_CAST(int64_t, s64, double, f64)
DEFINE_SVE_CAST(int32_t, s32, double, f64)
DEFINE_SVE_CAST(int16_t, s16, double, f64)
DEFINE_SVE_CAST(int64_t, s64, float, f32)
DEFINE_SVE_CAST(int32_t, s32, float, f32)
DEFINE_SVE_CAST(int16_t, s16, float, f32)
DEFINE_SVE_CAST(float, f32, double, f64)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<
        double>> inline gather(const double* base_addr, const Vectorized<int64_t>& vindex_) {
  svint64_t vindex =
      svasrd_n_s64_x(ptrue, svmul_s64_x(ptrue, vindex_, svdup_n_s64(scale)), 3);
  return svld1_gather_s64index_f64(ptrue, base_addr, vindex);
}

template <int64_t scale = 1>
std::enable_if_t<
    scale == 1 || scale == 2 || scale == 4 || scale == 8,
    Vectorized<
        float>> inline gather(const float* base_addr, const Vectorized<int32_t>& vindex_) {
  svint32_t vindex =
      svasrd_n_s32_x(ptrue, svmul_s32_x(ptrue, vindex_, svdup_n_s32(scale)), 2);
  return svld1_gather_s32index_f32(ptrue, base_addr, vindex);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MASK GATHER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <int64_t scale = 1>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<double>> inline mask_gather(
        const Vectorized<double>& src,
        const double* base_addr,
        const Vectorized<int64_t>& vindex_,
        const Vectorized<double>& mask_) {
  svbool_t mask =
      svcmpeq_s64(ptrue, svreinterpret_s64_f64(mask_), ALL_S64_TRUE_MASK);
  svint64_t vindex =
      svasrd_n_s64_x(ptrue, svmul_s64_x(ptrue, vindex_, svdup_n_s64(scale)), 3);
  return svsel_f64(
      mask, svld1_gather_s64index_f64(mask, base_addr, vindex), src);
}

template <int64_t scale = 1>
std::
    enable_if_t<scale == 1 || scale == 2 || scale == 4 || scale == 8, Vectorized<float>> inline mask_gather(
        const Vectorized<float>& src,
        const float* base_addr,
        const Vectorized<int32_t>& vindex_,
        const Vectorized<float>& mask_) {
  svbool_t mask =
      svcmpeq_s32(ptrue, svreinterpret_s32_f32(mask_), ALL_S32_TRUE_MASK);
  svint32_t vindex =
      svasrd_n_s32_x(ptrue, svmul_s32_x(ptrue, vindex_, svdup_n_s32(scale)), 2);
  return svsel_f32(
      mask, svld1_gather_s32index_f32(mask, base_addr, vindex), src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Only works for inputs in the range: [-2^51, 2^51]
// From: https://stackoverflow.com/a/41148578
template <>
Vectorized<int64_t> inline convert_to_int_of_same_size<double>(
    const Vectorized<double>& src) {
  svfloat64_t x = svadd_f64_x(ptrue, src, svdup_n_f64(0x0018000000000000));
  return svsub_s64_x(
      ptrue,
      svreinterpret_s64_f64(x),
      svreinterpret_s64_f64(svdup_n_f64(0x0018000000000000)));
}

template <>
Vectorized<int32_t> inline convert_to_int_of_same_size<float>(
    const Vectorized<float>& src) {
  return svcvt_s32_f32_x(ptrue, src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline interleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, a1, a3, a3}
  //   b = {b0, b1, b2, b3}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1}
  //          {a2, b2, a3, b3}
  return std::make_pair(
      Vectorized<double>(svzip1_f64(a, b)),
      Vectorized<double>(svzip2_f64(a, b)));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline interleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  return std::make_pair(
      Vectorized<float>(svzip1_f32(a, b)), Vectorized<float>(svzip2_f32(a, b)));
}

#ifdef __ARM_FEATURE_BF16
template <>
std::pair<
    Vectorized<c10::BFloat16>,
    Vectorized<c10::BFloat16>> inline interleave2<c10::
                                                      BFloat16>(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  // inputs:
  //   a = {a0, a1, a2, a3, a4, a5, a6, a7}
  //   b = {b0, b1, b2, b3, b4, b5, b6, b7}
  // group cols crossing lanes:
  //   return {a0, b0, a1, b1, a2, b2, a3, b3}
  //          {a4, b4, a5, b5, a6, b6, a7, b7}
  svbfloat16_t aReg = a;
  svbfloat16_t bReg = b;
  Vectorized<c10::BFloat16> c = svzip1_bf16(aReg, bReg);
  Vectorized<c10::BFloat16> d = svzip2_bf16(aReg, bReg);
  return std::make_pair(c, d);
}
#endif // __ARM_FEATURE_BF16

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DEINTERLEAVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
std::pair<Vectorized<double>, Vectorized<double>> inline deinterleave2<double>(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1}
  //   b = {a2, b2, a3, b3}
  // swap lanes:
  //   return {a0, a1, a2, a3}
  //          {b0, b1, b2, b3}
  return std::make_pair(
      Vectorized<double>(svuzp1_f64(a, b)),
      Vectorized<double>(svuzp2_f64(a, b)));
}

template <>
std::pair<Vectorized<float>, Vectorized<float>> inline deinterleave2<float>(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}
  // swap lanes:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  return std::make_pair(
      Vectorized<float>(svuzp1_f32(a, b)), Vectorized<float>(svuzp2_f32(a, b)));
}

#ifdef __ARM_FEATURE_BF16
template <>
std::pair<
    Vectorized<c10::BFloat16>,
    Vectorized<c10::BFloat16>> inline deinterleave2<c10::
                                                        BFloat16>(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  // inputs:
  //   a = {a0, b0, a1, b1, a2, b2, a3, b3}
  //   b = {a4, b4, a5, b5, a6, b6, a7, b7}
  // swap lanes:
  //   return {a0, a1, a2, a3, a4, a5, a6, a7}
  //          {b0, b1, b2, b3, b4, b5, b6, b7}
  svbfloat16_t aReg = a;
  svbfloat16_t bReg = b;
  Vectorized<c10::BFloat16> c = svuzp1_bf16(aReg, bReg);
  Vectorized<c10::BFloat16> d = svuzp2_bf16(aReg, bReg);
  return std::make_pair(c, d);
}
#endif // __ARM_FEATURE_BF16

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLIP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#define DEFINE_FLIP_FUNC(type, sve_func)                    \
  inline Vectorized<type> flip(const Vectorized<type>& v) { \
    return Vectorized<type>(sve_func(v));                   \
  }
// Use the macro to define the flip functions
DEFINE_FLIP_FUNC(float, svrev_f32)
DEFINE_FLIP_FUNC(double, svrev_f64)
DEFINE_FLIP_FUNC(int64_t, svrev_s64)
DEFINE_FLIP_FUNC(int32_t, svrev_s32)
DEFINE_FLIP_FUNC(int16_t, svrev_s16)
DEFINE_FLIP_FUNC(int8_t, svrev_s8)

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
