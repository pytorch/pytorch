#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with SVE]

#include <ATen/cpu/vec/intrinsics.h>

#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>

#if defined(CPU_CAPABILITY_SVE256)
#include <ATen/cpu/vec/sve/vec_bfloat16.h>
#include <ATen/cpu/vec/sve/vec_double.h>
#include <ATen/cpu/vec/sve/vec_float.h>
#include <ATen/cpu/vec/sve/vec_int.h>
#include <ATen/cpu/vec/sve/vec_qint.h>
#endif

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

#if defined(CPU_CAPABILITY_SVE256)

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

#ifdef __ARM_FEATURE_BF16
DEFINE_SVE_CAST(int64_t, s64, c10::BFloat16, bf16)
DEFINE_SVE_CAST(int32_t, s32, c10::BFloat16, bf16)
DEFINE_SVE_CAST(int16_t, s16, c10::BFloat16, bf16)
#endif // __ARM_FEATURE_BF16

template <typename T>
inline constexpr bool is_sve_vec_bool_v =
    !std::is_same_v<VecBool<T>, Vectorized<T>>;

// Between types of the same width a comparison result stays a predicate, as
// the lanes line up. Otherwise it goes through its mask, like cast does.
template <typename dst_t, typename src_t>
inline auto cast(const SveVecBool<src_t>& p) {
  if constexpr (is_sve_vec_bool_v<dst_t> && sizeof(dst_t) == sizeof(src_t)) {
    return VecBool<dst_t>(svbool_t(p));
  } else {
    return cast<dst_t>(Vectorized<src_t>(p));
  }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ VECBOOL OPS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A comparison result in a bitwise op with a Vectorized<T> is used as its
// mask. Two comparison results combine as predicates (see SveVecBool).
#define DEFINE_SVE_VECBOOL_BINARY_OP(op)                                      \
  template <typename T>                                                       \
  inline Vectorized<T> operator op(SveVecBool<T> a, const Vectorized<T>& b) { \
    return Vectorized<T>(a) op b;                                             \
  }                                                                           \
  template <typename T>                                                       \
  inline Vectorized<T> operator op(const Vectorized<T>& a, SveVecBool<T> b) { \
    return a op Vectorized<T>(b);                                             \
  }

DEFINE_SVE_VECBOOL_BINARY_OP(&)
DEFINE_SVE_VECBOOL_BINARY_OP(|)
DEFINE_SVE_VECBOOL_BINARY_OP(^)

#undef DEFINE_SVE_VECBOOL_BINARY_OP

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
  svbool_t mask = svcmplt_s64(ptrue, svreinterpret_s64_f64(mask_), ZERO_S64);
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
  svbool_t mask = svcmplt_s32(ptrue, svreinterpret_s32_f32(mask_), ZERO_S32);
  svint32_t vindex =
      svasrd_n_s32_x(ptrue, svmul_s32_x(ptrue, vindex_, svdup_n_s32(scale)), 2);
  return svsel_f32(
      mask, svld1_gather_s32index_f32(mask, base_addr, vindex), src);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONVERT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

template <>
Vectorized<int64_t> inline convert_to_int_of_same_size<double>(
    const Vectorized<double>& src) {
  return svcvt_s64_f64_x(ptrue, src);
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
  return std::make_pair(
      Vectorized<c10::BFloat16>(svzip1_bf16(a, b)),
      Vectorized<c10::BFloat16>(svzip2_bf16(a, b)));
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
  return std::make_pair(
      Vectorized<c10::BFloat16>(svuzp1_bf16((svbfloat16_t)a, (svbfloat16_t)b)),
      Vectorized<c10::BFloat16>(svuzp2_bf16((svbfloat16_t)a, (svbfloat16_t)b)));
}
#endif // __ARM_FEATURE_BF16

#endif // defined(CPU_CAPABILITY_SVE256)

} // namespace CPU_CAPABILITY
} // namespace at::vec
