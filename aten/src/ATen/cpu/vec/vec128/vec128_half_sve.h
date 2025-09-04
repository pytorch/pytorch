#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec128/vec128_reduced_precision_common_neon.h>
#include <c10/util/Half.h>
#include <c10/util/irange.h>

#include <arm_neon.h>
#include <arm_neon_sve_bridge.h>
#include <arm_sve.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

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
  Vectorized(float16_t val) : Vectorized16(svget_neonq(svdup_n_f16(val))) {}
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

  Vectorized(svfloat16_t v) : Vectorized16(svget_neonq(v)) {}
  template <
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ float16_t buffer[size()] = {vals...};
    values = vld1q_f16(buffer);
  }
  operator svfloat16_t() const {
    return svset_neonq(svundef_f16(), values);
  }
  svfloat16_t valuesAsSve() const {
    return svset_neonq(svundef_f16(), values);
  }

  template <int64_t mask>
  static Vectorized<c10::Half> blend(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint16x8_t maskArray = {
        (mask & 1ULL) ? 0xFFFF : 0,
        (mask & 2ULL) ? 0xFFFF : 0,
        (mask & 4ULL) ? 0xFFFF : 0,
        (mask & 8ULL) ? 0xFFFF : 0,
        (mask & 16ULL) ? 0xFFFF : 0,
        (mask & 32ULL) ? 0xFFFF : 0,
        (mask & 64ULL) ? 0xFFFF : 0,
        (mask & 128ULL) ? 0xFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_f16(maskArray, b.values, a.values);
  }

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
    return vbslq_f16(vreinterpretq_u16_f16(mask.values), b.values, a.values);
  }
  static Vectorized<c10::Half> set(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f16(svwhilelt_b16(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<c10::Half> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
    }
    svbool_t pg = svwhilelt_b16(0ull, count);
    return svld1_f16(pg, reinterpret_cast<const float16_t*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f16(reinterpret_cast<float16_t*>(ptr), values);
      return;
    } else {
      svbool_t pg = svwhilelt_b16(0ull, count);
      svst1_f16(pg, reinterpret_cast<float16_t*>(ptr), valuesAsSve());
    }
  }
  inline c10::Half operator[](int idx) const {
    return values[idx];
  }
  inline c10::Half operator[](int idx) {
    return values[idx];
  }
  int zero_mask() const {
    uint16x8_t cmpReg = vceqzq_f16(values);
    uint8x8_t narrowedCmp = vmovn_u16(cmpReg);
    uint64x2_t extReg = svget_neonq(svbext_u64(
        svset_neonq(
            svundef_u64(),
            vreinterpretq_u64_u8(vcombine_u8(narrowedCmp, vdup_n_u8(0)))),
        svreinterpret_u64_u8(svdup_u8(1))));
    return extReg[0];
  }
  Vectorized<c10::Half> isnan() const {
    return vreinterpretq_f16_u16(vmvnq_u16(vceqq_f16(values, values)));
  }
  bool has_inf_nan() const {
    return svptest_any(
        ptrue,
        svcmpuo_f16(
            ptrue,
            svset_neonq(svundef_f16(), vsubq_f16(values, values)),
            ZERO_F16));
  }
  Vectorized<c10::Half> ceil() const {
    return vrndpq_f16(values);
  }
  Vectorized<c10::Half> floor() const {
    return vrndmq_f16(values);
  }
  Vectorized<c10::Half> round() const {
    return vrndiq_f16(values);
  }
  Vectorized<c10::Half> abs() const {
    return Vectorized<c10::Half>(vabsq_f16(values));
  }
  Vectorized<c10::Half> frac() const;
  Vectorized<c10::Half> neg() const {
    return Vectorized<c10::Half>(vnegq_f16(values));
  }
  Vectorized<c10::Half> trunc() const {
    return Vectorized<c10::Half>(vrndq_f16(values));
  }
  Vectorized<c10::Half> sqrt() const {
    return Vectorized<c10::Half>(vsqrtq_f16(values));
  }
  Vectorized<c10::Half> reciprocal() const {
    float16x8_t recip = vrecpeq_f16(values);
    recip = vmulq_f16(vrecpsq_f16(values, recip), recip);
    recip = vmulq_f16(vrecpsq_f16(values, recip), recip);
    return recip;
  }
  Vectorized<c10::Half> rsqrt() const {
    float16x8_t sqrt_reciprocal = vrsqrteq_f16(values);
    sqrt_reciprocal = vmulq_f16(
        vrsqrtsq_f16(vmulq_f16(values, sqrt_reciprocal), sqrt_reciprocal),
        sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f16(
        vrsqrtsq_f16(vmulq_f16(values, sqrt_reciprocal), sqrt_reciprocal),
        sqrt_reciprocal);

    return sqrt_reciprocal;
  }
  c10::Half reduce_add() const {
    return svaddv_f16(ptrue, svset_neonq(svundef_f16(), values));
  }
  c10::Half reduce_max() const {
    return svmaxv_f16(ptrue, svset_neonq(svundef_f16(), values));
  }
  Vectorized<c10::Half> operator==(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vceqq_f16(values, other.values)));
  }

  Vectorized<c10::Half> operator!=(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vmvnq_u16(vceqq_f16(values, other.values))));
  }

  Vectorized<c10::Half> operator<(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcltq_f16(values, other.values)));
  }

  Vectorized<c10::Half> operator<=(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcleq_f16(values, other.values)));
  }

  Vectorized<c10::Half> operator>(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcgtq_f16(values, other.values)));
  }

  Vectorized<c10::Half> operator>=(const Vectorized<c10::Half>& other) const {
    return Vectorized<c10::Half>(
        vreinterpretq_f16_u16(vcgeq_f16(values, other.values)));
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
  return Vectorized<c10::Half>(vaddq_f16(a, b));
}

inline void load_fp32_from_fp16(const c10::Half* data, Vectorized<float>& out) {
  const __fp16* dataPtr = reinterpret_cast<const __fp16*>(data);
  float16x4_t lowf16 = vld1_f16(dataPtr);
  out = vcvt_f32_f16(lowf16);
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
  return Vectorized<c10::Half>(vsubq_f16(a, b));
}

template <>
Vectorized<c10::Half> inline operator*(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vmulq_f16(a, b));
}

template <>
Vectorized<c10::Half> inline operator/(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vdivq_f16(a, b));
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
  return Vectorized<c10::Half>(vmaxq_f16(a, b));
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<c10::Half> inline minimum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  return Vectorized<c10::Half>(vminq_f16(a, b));
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
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this == other), 1));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ne(
    const Vectorized<c10::Half>& other) const {
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this != other), 1));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::gt(
    const Vectorized<c10::Half>& other) const {
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this > other), 1));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::ge(
    const Vectorized<c10::Half>& other) const {
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this >= other), 1));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::lt(
    const Vectorized<c10::Half>& other) const {
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this < other), 1));
}

inline Vectorized<c10::Half> Vectorized<c10::Half>::le(
    const Vectorized<c10::Half>& other) const {
  return svreinterpret_f16_u16(
      svand_n_u16_x(ptrue, svreinterpret_u16_f16(*this <= other), 1));
}

template <>
inline void convert(const float16_t* src, int16_t* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<c10::Half>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_f16(src);
    auto vec2 = vld1q_f16(src + oneRegElemCount);
    auto vec3 = vld1q_f16(src + twoRegsElemCount);
    auto vec4 = vld1q_f16(src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = vcvtq_s16_f16(vec1);
    auto convertedVec2 = vcvtq_s16_f16(vec2);
    auto convertedVec3 = vcvtq_s16_f16(vec3);
    auto convertedVec4 = vcvtq_s16_f16(vec4);
    vst1q_s16(dst, convertedVec1);
    vst1q_s16(dst + oneRegElemCount, convertedVec2);
    vst1q_s16(dst + twoRegsElemCount, convertedVec3);
    vst1q_s16(dst + (twoRegsElemCount + oneRegElemCount), convertedVec4);
    src += fourRegsElemCount;
    dst += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0;
       --remainder) {
    *dst = *src;
    src += 1;
    dst += 1;
  }
}

template <>
inline void convert(const int16_t* src, float16_t* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<c10::Half>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_s16(src);
    auto vec2 = vld1q_s16(src + oneRegElemCount);
    auto vec3 = vld1q_s16(src + twoRegsElemCount);
    auto vec4 = vld1q_s16(src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = vcvtq_f16_s16(vec1);
    auto convertedVec2 = vcvtq_f16_s16(vec2);
    auto convertedVec3 = vcvtq_f16_s16(vec3);
    auto convertedVec4 = vcvtq_f16_s16(vec4);
    vst1q_f16(dst, convertedVec1);
    vst1q_f16(dst + oneRegElemCount, convertedVec2);
    vst1q_f16(dst + twoRegsElemCount, convertedVec3);
    vst1q_f16(dst + (twoRegsElemCount + oneRegElemCount), convertedVec4);
    src += fourRegsElemCount;
    dst += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0;
       --remainder) {
    *dst = *src;
    src += 1;
    dst += 1;
  }
}

template <>
Vectorized<c10::Half> inline fmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  return Vectorized<c10::Half>(vfmaq_f16(c, a, b));
}

template <>
Vectorized<c10::Half> inline fnmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  return Vectorized<c10::Half>(vfmsq_f16(c, a, b));
}

template <>
Vectorized<c10::Half> inline fmsub(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  return Vectorized<c10::Half>(svnmsb_f16_x(ptrue, a, b, c));
}

template <>
Vectorized<c10::Half> inline fnmsub(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  return Vectorized<c10::Half>(svnmad_f16_x(ptrue, a, b, c));
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
