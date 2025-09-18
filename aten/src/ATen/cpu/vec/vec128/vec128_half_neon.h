#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_reduced_precision_common_neon.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Right now contains only aarch64 implementation.
// Due to follow two reasons aarch32 is not currently supported.
// 1. Due to difference in ISA been aarch32 and aarch64, intrinsics
//    that work for aarch64 dont work for aarch32.
// 2. Android NDK r21 has problems with compiling aarch32.
//    Clang seg faults.
//    https://github.com/android/ndk/issues/1248
//    https://bugs.llvm.org/show_bug.cgi?id=45824
// Most likely we will do aarch32 support with inline asm.
#if !defined(C10_MOBILE) && defined(__aarch64__)
#ifdef __BIG_ENDIAN__
#error "Big endian is not supported."
#endif

template <int index, bool mask_val>
struct BlendHalfRegs {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res);
};

template <int index>
struct BlendHalfRegs<index, true> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    return vsetq_lane_f16(vgetq_lane_f16(b, index), res, index);
  }
};

template <int index>
struct BlendHalfRegs<index, false> {
  static float16x8_t impl(
      const float16x8_t& a,
      const float16x8_t& b,
      float16x8_t& res) {
    return vsetq_lane_f16(vgetq_lane_f16(a, index), res, index);
  }
};

template <>
struct is_vec_specialized_for<c10::Half> : std::bool_constant<true> {};

// On ARM, Half type supports float16_t->Half constructor and Half->float16_t
// conversion
template <>
class Vectorized<c10::Half> : public Vectorized16<
                                  float16x8_t,
                                  c10::Half,
                                  BlendHalfRegs,
                                  Vectorized<c10::Half>> {
  using Base = Vectorized16<
      float16x8_t,
      c10::Half,
      BlendHalfRegs,
      Vectorized<c10::Half>>;
  friend Base;

 private:
  // We use these private map functions to implement various methods
  Vectorized<c10::Half> map_with_vec_float_method(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)();
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)();
    float16x4_t r00 = vcvt_f16_f32(mv0);
    float16x4_t r01 = vcvt_f16_f32(mv1);
    return Vectorized<c10::Half>(vcombine_f16(r00, r01));
  }

  Vectorized<c10::Half> map2_with_vec_float_method(
      const Vectorized<c10::Half>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values));
    float32x4_t second_v00 = vcvt_f32_f16(vget_low_f16(second.values));
    float32x4_t second_v01 = vcvt_f32_f16(vget_high_f16(second.values));
    Vectorized<float> mv0 =
        (Vectorized<float>(v00).*m)(Vectorized<float>(second_v00));
    Vectorized<float> mv1 =
        (Vectorized<float>(v01).*m)(Vectorized<float>(second_v01));
    float16x4_t r00 = vcvt_f16_f32(mv0);
    float16x4_t r01 = vcvt_f16_f32(mv1);

    // Pack result into Vectorized<c10::Half>
    return Vectorized<c10::Half>(vcombine_f16(r00, r01));
  }

  Vectorized<c10::Half> map2_bitmask_with_vec_float_method(
      const Vectorized<c10::Half>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values));
    float32x4_t second_v00 = vcvt_f32_f16(vget_low_f16(second.values));
    float32x4_t second_v01 = vcvt_f32_f16(vget_high_f16(second.values));
    Vectorized<float> mv0 =
        (Vectorized<float>(v00).*m)(Vectorized<float>(second_v00));
    Vectorized<float> mv1 =
        (Vectorized<float>(v01).*m)(Vectorized<float>(second_v01));
    // Assume the operator returns a bitmask, not "real" floats, and
    // just narrow the bits. All-ones is a NaN and will get mangled by
    // conversion!
    float16x4_t r00 =
        vreinterpret_f16_u16(vmovn_u32(vreinterpretq_u32_f32(mv0)));
    float16x4_t r01 =
        vreinterpret_f16_u16(vmovn_u32(vreinterpretq_u32_f32(mv1)));

    // Pack result into Vectorized<c10::Half>
    return Vectorized<c10::Half>(vcombine_f16(r00, r01));
  }

 public:
  using Vectorized16::Vectorized16;

  Vectorized() = default;

  // A ctor that accepts c10::Half is needed to fit interface with vec_base.h
  // A second constructor that takes float16_t is also included
  Vectorized(c10::Half val) : Vectorized((float16_t)val) {}
  Vectorized(float16_t val) : Vectorized16(vdupq_n_f16(val)) {}
  Vectorized(
      value_type val0,
      value_type val1,
      value_type val2,
      value_type val3,
      value_type val4,
      value_type val5,
      value_type val6,
      value_type val7)
      : Vectorized16(
            float16x8_t{val0, val1, val2, val3, val4, val5, val6, val7}) {}

  static Vectorized<c10::Half> blendv(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      const Vectorized<c10::Half>& mask) {
    // Note: using blendv is very awkward because 0xFFFF is one of
    // many NaN's in FP16 It's unfortunate that the mask has type Half
    // (required from vec_base)

    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.

    // NOTE [vbslq_f16]: vbslq_f16 doesn't work on clang without
    // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC. vbslq_u16 generates the
    // same instruction anyway. see https://godbolt.org/z/cY4a55Y7P
    Vectorized<c10::Half> vec(mask.values);
    vec.values = vreinterpretq_f16_u16(vbslq_u16(
        vreinterpretq_u16_f16(vec.values),
        vreinterpretq_u16_f16(b.values),
        vreinterpretq_u16_f16(a.values)));
    return vec;
  }
  static Vectorized<c10::Half> set(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      int64_t count = size()) {
    uint16_t pre_mask[size()] = {0};
    for (int i = 0; i < count; i++) {
      pre_mask[i] = 0xFFFF;
    }
    uint16x8_t mask = vld1q_u16(pre_mask);

    // Using blendv is awkward because 0xFFFF is one of many NaN's in FP16
    // so we directly use vbslq_u16 instead. (See NOTE [vbslq_f16] above.)
    Vectorized<c10::Half> vec(vreinterpretq_f16_u16(vbslq_u16(
        mask,
        vreinterpretq_u16_f16(b.values),
        vreinterpretq_u16_f16(a.values))));

    return vec;
  }
  static Vectorized<c10::Half> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
    }
    __at_align__ float16_t tmp_values[size()];
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float16_t*>(ptr),
        count * sizeof(float16_t));
    return vld1q_f16(reinterpret_cast<const float16_t*>(tmp_values));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f16(reinterpret_cast<float16_t*>(ptr), values);
      return;
    } else {
      float16_t tmp_values[size()];
      vst1q_f16(reinterpret_cast<float16_t*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float16_t));
    }
  }
  int zero_mask() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    uint16x8_t is_zero_vec = vceqzq_f16(values);
    const int16x8_t shift = vcombine_s16(
        vcreate_s16(
            0x0 | (int64_t(0x1) << 16) | (int64_t(0x2) << 32) |
            (int64_t(0x3) << 48)),
        vcreate_s16(
            0x4 | (int64_t(0x5) << 16) | (int64_t(0x6) << 32) |
            (int64_t(0x7) << 48)));
    uint16x8_t bits_vec =
        vshlq_u16(vandq_u16(is_zero_vec, vdupq_n_u16(1)), shift);
    return vaddvq_u16(bits_vec);
#else // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // use known working implmentation.
    __at_align__ value_type tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  }
  Vectorized<c10::Half> isnan() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return vreinterpretq_f16_u16(vmvnq_u16(vceqq_f16(values, values)));
#else
    // NOTE: we could make this faster by doing vectorized checks of
    // exponent/payload bits.
    __at_align__ c10::Half tmp[size()];
    __at_align__ c10::Half res[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i])) {
        std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(c10::Half));
      } else {
        std::memset(static_cast<void*>(&res[i]), 0, sizeof(c10::Half));
      }
    }
    return loadu(res);
#endif
  }
  bool has_inf_nan() const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }
  Vectorized<c10::Half> abs() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(vabsq_f16(values));
#else
    return map_with_vec_float_method(&Vectorized<float>::abs);
#endif
  }
  Vectorized<c10::Half> frac() const;
  Vectorized<c10::Half> neg() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(vnegq_f16(values));
#else
    return map_with_vec_float_method(&Vectorized<float>::neg);
#endif
  }
  Vectorized<c10::Half> trunc() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(vrndq_f16(values));
#else
    return map_with_vec_float_method(&Vectorized<float>::trunc);
#endif
  }
  Vectorized<c10::Half> sqrt() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(vsqrtq_f16(values));
#else
    return map_with_vec_float_method(&Vectorized<float>::sqrt);
#endif
  }
  Vectorized<c10::Half> reciprocal() const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    auto ones = vdupq_n_f16(1.0f);
    return Vectorized<c10::Half>(vdivq_f16(ones, values));
#else
    return map_with_vec_float_method(&Vectorized<float>::reciprocal);
#endif
  }
  Vectorized<c10::Half> operator==(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vceqq_f16(values, other.values)));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator==);
#endif
  }

  Vectorized<c10::Half> operator!=(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vmvnq_u16(vceqq_f16(values, other.values))));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator!=);
#endif
  }

  Vectorized<c10::Half> operator<(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcltq_f16(values, other.values)));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator<);
#endif
  }

  Vectorized<c10::Half> operator<=(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcleq_f16(values, other.values)));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator<=);
#endif
  }

  Vectorized<c10::Half> operator>(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcgtq_f16(values, other.values)));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator>);
#endif
  }

  Vectorized<c10::Half> operator>=(const Vectorized<c10::Half>& other) const {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcgeq_f16(values, other.values)));
#else
    return map2_bitmask_with_vec_float_method(
        other, &Vectorized<float>::operator>=);
#endif
  }

  Vectorized<c10::Half> eq(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> ne(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> gt(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> ge(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> lt(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> le(const Vectorized<c10::Half>& other) const;
}; // Vectorized<Half>

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_half_float(
    const Vectorized<Half>& a) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  float16x8_t x = a;
  float32x4_t x1 = vcvt_f32_f16(vget_low_f16(x));
  float32x4_t x2 = vcvt_f32_f16(vget_high_f16(x));
  return {Vectorized<float>(x1), Vectorized<float>(x2)};
}
inline Vectorized<Half> convert_float_half(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  static_assert(Vectorized<Half>::size() == 2 * Vectorized<float>::size());
  float32x4_t x = a;
  float32x4_t y = b;
  float16x4_t x1 = vcvt_f16_f32(x);
  float16x4_t x2 = vcvt_f16_f32(y);
  return Vectorized<Half>(vcombine_f16(x1, x2));
}

template <typename Op>
Vectorized<c10::Half> binary_operator_via_float(
    Op op,
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  const auto [a_float_low, a_float_high] = convert_half_float(a);
  const auto [b_float_low, b_float_high] = convert_half_float(b);
  return convert_float_half(
      op(a_float_low, b_float_low), op(a_float_high, b_float_high));
}

template <>
Vectorized<c10::Half> inline operator+(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vaddq_f16(a, b));
#else
  return binary_operator_via_float(std::plus<Vectorized<float>>(), a, b);
#endif
}

inline void load_fp32_from_fp16(const c10::Half* data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()];
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k];
  }
  out = Vectorized<float>::loadu(values);
}

inline void load_fp32_from_fp16(
    const c10::Half* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  Vectorized<c10::Half> f16_vec = Vectorized<c10::Half>::loadu(data);
  auto floats = convert_half_float(f16_vec);
  out1 = std::get<0>(floats);
  out2 = std::get<1>(floats);
}

template <>
Vectorized<c10::Half> inline operator-(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vsubq_f16(a, b));
#else
  return binary_operator_via_float(std::minus<Vectorized<float>>(), a, b);
#endif
}

template <>
Vectorized<c10::Half> inline operator*(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vmulq_f16(a, b));
#else
  return binary_operator_via_float(std::multiplies<Vectorized<float>>(), a, b);
#endif
}

template <>
Vectorized<c10::Half> inline operator/(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vdivq_f16(a, b));
#else
  return binary_operator_via_float(std::divides<Vectorized<float>>(), a, b);
#endif
}

// frac. Implement this here so we can use subtraction
inline Vectorized<c10::Half> Vectorized<c10::Half>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<c10::Half> inline maximum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vmaxq_f16(a, b));
#else
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&maximum),
      a,
      b);
#endif
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<c10::Half> inline minimum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vminq_f16(a, b));
#else
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&minimum),
      a,
      b);
#endif
}

template <>
Vectorized<c10::Half> inline clamp(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min,
    const Vectorized<c10::Half>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<c10::Half> inline clamp_max(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& max) {
  return minimum(max, a);
}

template <>
Vectorized<c10::Half> inline clamp_min(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& min) {
  return maximum(min, a);
}

template <>
Vectorized<c10::Half> inline operator&(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vreinterpretq_f16_u16(
      vandq_u16(vreinterpretq_u16_f16(a), vreinterpretq_u16_f16(b))));
}

template <>
Vectorized<c10::Half> inline operator|(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vreinterpretq_f16_u16(
      vorrq_u16(vreinterpretq_u16_f16(a), vreinterpretq_u16_f16(b))));
}

template <>
Vectorized<c10::Half> inline operator^(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vreinterpretq_f16_u16(
      veorq_u16(vreinterpretq_u16_f16(a), vreinterpretq_u16_f16(b))));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::eq(
    const Vectorized<c10::Half>& other) const {
  return (*this == other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ne(
    const Vectorized<c10::Half>& other) const {
  return (*this != other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::gt(
    const Vectorized<c10::Half>& other) const {
  return (*this > other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ge(
    const Vectorized<c10::Half>& other) const {
  return (*this >= other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::lt(
    const Vectorized<c10::Half>& other) const {
  return (*this < other) & Vectorized<c10::Half>(1);
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::le(
    const Vectorized<c10::Half>& other) const {
  return (*this <= other) & Vectorized<c10::Half>(1);
}

// These are global functions, so the defaults in vec_base.h should
// work fine if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC is not available.
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline void convert(const float16_t* src, int16_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    vst1q_s16(dst + i, vcvtq_s16_f16(vld1q_f16(src + i)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<int16_t>(src[i]);
  }
}

template <>
inline void convert(const int16_t* src, float16_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    vst1q_f16(dst + i, vcvtq_f16_s16(vld1q_s16(src + i)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float16_t>(src[i]);
  }
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <>
Vectorized<c10::Half> inline fmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vfmaq_f16(c, a, b));
#else
  return a * b + c;
#endif
}

template <>
Vectorized<c10::Half> inline fnmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vfmsq_f16(c, a, b));
#else
  return -a * b + c;
#endif
}

template <>
Vectorized<c10::Half> inline fmsub(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vnegq_f16(vfmsq_f16(c, a, b)));
#else
  return a * b - c;
#endif
}

template <>
Vectorized<c10::Half> inline fnmsub(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  return Vectorized<c10::Half>(vnegq_f16(vfmaq_f16(c, a, b)));
#else
  return -a * b - c;
#endif
}

#else

#define CONVERT_NON_VECTORIZED_INIT(type, name)                     \
  inline std::tuple<Vectorized<float>, Vectorized<float>>           \
      convert_##name##_float(const Vectorized<type>& a) {           \
    constexpr int64_t K = Vectorized<type>::size();                 \
    __at_align__ float arr[K];                                      \
    __at_align__ type arr2[K];                                      \
    a.store(arr2);                                                  \
    convert(arr2, arr, K);                                          \
    return std::make_tuple(                                         \
        Vectorized<float>::loadu(arr),                              \
        Vectorized<float>::loadu(arr + Vectorized<float>::size())); \
  }                                                                 \
  inline Vectorized<type> convert_float_##name(                     \
      const Vectorized<float>& a, const Vectorized<float>& b) {     \
    constexpr int64_t K = Vectorized<type>::size();                 \
    __at_align__ float arr[K];                                      \
    __at_align__ type arr2[K];                                      \
    a.store(arr);                                                   \
    b.store(arr + Vectorized<float>::size());                       \
    convert(arr, arr2, K);                                          \
    return Vectorized<type>::loadu(arr2);                           \
  }

#define LOAD_FP32_NON_VECTORIZED_INIT(type, name)                           \
  inline void load_fp32_from_##name(                                        \
      const type* data, Vectorized<float>& out) {                           \
    __at_align__ float values[Vectorized<float>::size()];                   \
    for (const auto k : c10::irange(Vectorized<float>::size())) {           \
      values[k] = data[k];                                                  \
    }                                                                       \
    out = Vectorized<float>::loadu(values);                                 \
  }                                                                         \
                                                                            \
  inline void load_fp32_from_##name(                                        \
      const type* data, Vectorized<float>& out1, Vectorized<float>& out2) { \
    load_fp32_from_##name(data, out1);                                      \
    data += Vectorized<float>::size();                                      \
    load_fp32_from_##name(data, out2);                                      \
  }

CONVERT_NON_VECTORIZED_INIT(Half, half)

LOAD_FP32_NON_VECTORIZED_INIT(Half, fp16)

#endif // !defined(C10_MOBILE) && defined(__aarch64__)

} // namespace CPU_CAPABILITY
} // namespace at::vec
