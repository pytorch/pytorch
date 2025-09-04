#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec128/vec128_reduced_precision_common_neon.h>
#include <c10/util/BFloat16.h>
#include <c10/util/bit_cast.h>
#include <c10/util/irange.h>

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

// Following vec128_half_neon.h, we only support aarch64.
#if !defined(C10_MOBILE) && defined(__aarch64__)
#ifdef __BIG_ENDIAN__
#error "Big endian is not supported."
#endif

template <int index, bool mask_val>
struct BlendBFloat16Regs {
  static bfloat16x8_t impl(
      const bfloat16x8_t& a,
      const bfloat16x8_t& b,
      bfloat16x8_t& res);
};

template <int index>
struct BlendBFloat16Regs<index, true> {
  static bfloat16x8_t impl(
      const bfloat16x8_t& a,
      const bfloat16x8_t& b,
      bfloat16x8_t& res) {
    return vsetq_lane_bf16(vgetq_lane_bf16(b, index), res, index);
  }
};

template <int index>
struct BlendBFloat16Regs<index, false> {
  static bfloat16x8_t impl(
      const bfloat16x8_t& a,
      const bfloat16x8_t& b,
      bfloat16x8_t& res) {
    return vsetq_lane_bf16(vgetq_lane_bf16(a, index), res, index);
  }
};

template <>
struct is_vec_specialized_for<c10::BFloat16> : std::bool_constant<true> {};

template <>
class Vectorized<c10::BFloat16> : public Vectorized16<
                                      bfloat16x8_t,
                                      c10::BFloat16,
                                      BlendBFloat16Regs,
                                      Vectorized<c10::BFloat16>> {
  using Base = Vectorized16<
      bfloat16x8_t,
      c10::BFloat16,
      BlendBFloat16Regs,
      Vectorized<c10::BFloat16>>;
  friend Base;
  friend std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
      const Vectorized<c10::BFloat16>& a);
  friend Vectorized<c10::BFloat16> convert_float_bfloat16(
      const Vectorized<float>& a,
      const Vectorized<float>& b);

 private:
  Vectorized<c10::BFloat16> map2(
      const Vectorized<c10::BFloat16>& second,
      c10::BFloat16 (*const f)(c10::BFloat16, c10::BFloat16)) const {
    bfloat16x8_t result;
    result[0] = f(values[0], second[0]);
    result[1] = f(values[1], second[1]);
    result[2] = f(values[2], second[2]);
    result[3] = f(values[3], second[3]);
    result[4] = f(values[4], second[4]);
    result[5] = f(values[5], second[5]);
    result[6] = f(values[6], second[6]);
    result[7] = f(values[7], second[7]);
    return result;
  }

  static float32x4_t convert_f32_bf16(bfloat16x4_t bf16) {
    return vcvt_f32_bf16(bf16);
  }

  static bfloat16x4_t convert_bf16_f32(const Vectorized<float>& f32) {
    return vcvt_bf16_f32(f32);
  }

  Vectorized<c10::BFloat16> map_with_vec_float_method(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    float32x4_t v00 = convert_f32_bf16(vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(vget_high_bf16(values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)();
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)();
    bfloat16x4_t r00 = convert_bf16_f32(mv0);
    bfloat16x4_t r01 = convert_bf16_f32(mv1);
    return Vectorized<c10::BFloat16>(vcombine_bf16(r00, r01));
  }

  Vectorized<c10::BFloat16> map2_with_vec_float_method(
      const Vectorized<c10::BFloat16>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = convert_f32_bf16(vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(vget_high_bf16(values));
    float32x4_t second_v00 = convert_f32_bf16(vget_low_bf16(second.values));
    float32x4_t second_v01 = convert_f32_bf16(vget_high_bf16(second.values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)(second_v00);
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)(second_v01);
    bfloat16x4_t r00 = convert_bf16_f32(mv0);
    bfloat16x4_t r01 = convert_bf16_f32(mv1);
    return Vectorized<c10::BFloat16>(vcombine_bf16(r00, r01));
  }

  Vectorized<c10::BFloat16> map2_bitmask_with_vec_float_method(
      const Vectorized<c10::BFloat16>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = convert_f32_bf16(vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(vget_high_bf16(values));
    float32x4_t second_v00 = convert_f32_bf16(vget_low_bf16(second.values));
    float32x4_t second_v01 = convert_f32_bf16(vget_high_bf16(second.values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)(second_v00);
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)(second_v01);
    // Assume the operator returns a bitmask, not "real" floats, and
    // just narrow the bits. All-ones is a NaN and will get mangled by
    // conversion!
    bfloat16x4_t r00 =
        vreinterpret_bf16_u16(vmovn_u32(vreinterpretq_u32_f32(mv0)));
    bfloat16x4_t r01 =
        vreinterpret_bf16_u16(vmovn_u32(vreinterpretq_u32_f32(mv1)));
    return Vectorized<c10::BFloat16>(vcombine_bf16(r00, r01));
  }

 public:
  using Vectorized16::Vectorized16;

  Vectorized() = default;
  Vectorized(svbfloat16_t v) : Vectorized16(svget_neonq(v)) {}
  Vectorized(c10::BFloat16 val)
      : Vectorized16(svget_neonq(svdup_n_bf16(c10::bit_cast<bfloat16_t>(val.x)))) {}
  Vectorized(float val) : Vectorized(c10::BFloat16(val)) {}
  Vectorized(
      value_type val0,
      value_type val1,
      value_type val2,
      value_type val3,
      value_type val4,
      value_type val5,
      value_type val6,
      value_type val7)
      : Vectorized16(bfloat16x8_t{
            c10::bit_cast<bfloat16_t>(val0.x),
            c10::bit_cast<bfloat16_t>(val1.x),
            c10::bit_cast<bfloat16_t>(val2.x),
            c10::bit_cast<bfloat16_t>(val3.x),
            c10::bit_cast<bfloat16_t>(val4.x),
            c10::bit_cast<bfloat16_t>(val5.x),
            c10::bit_cast<bfloat16_t>(val6.x),
            c10::bit_cast<bfloat16_t>(val7.x)}) {}
  operator svbfloat16_t() const {
    return svset_neonq(svundef_bf16(), values);
  }
  svbfloat16_t valuesAsSve() const {
    return svset_neonq(svundef_bf16(), values);
  }

  template <int64_t mask>
  static Vectorized<c10::BFloat16> blend(
      const Vectorized<c10::BFloat16>& a,
      const Vectorized<c10::BFloat16>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding bit in
    // 'mask' is set, 0 otherwise.
    uint16x8_t maskArray = {
      (mask &   1ULL) ? 0xFFFF : 0, 
      (mask &   2ULL) ? 0xFFFF : 0, 
      (mask &   4ULL) ? 0xFFFF : 0, 
      (mask &   8ULL) ? 0xFFFF : 0,
      (mask &  16ULL) ? 0xFFFF : 0, 
      (mask &  32ULL) ? 0xFFFF : 0, 
      (mask &  64ULL) ? 0xFFFF : 0, 
      (mask & 128ULL) ? 0xFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_f16(maskArray, b.values, a.values);
  }

  static Vectorized<c10::BFloat16> blendv(
      const Vectorized<c10::BFloat16>& a,
      const Vectorized<c10::BFloat16>& b,
      const Vectorized<c10::BFloat16>& mask) {
    // NOTE: blendv has the same problems as it does for Half; see comments in
    // vec128_half_neon.h.
    Vectorized<c10::BFloat16> vec(mask.values);
    vec.values = vreinterpretq_bf16_u16(vbslq_u16(
        vreinterpretq_u16_bf16(vec.values),
        vreinterpretq_u16_bf16(b.values),
        vreinterpretq_u16_bf16(a.values)));
    return vec;
  }
  static Vectorized<c10::BFloat16> set(
      const Vectorized<c10::BFloat16>& a,
      const Vectorized<c10::BFloat16>& b,
      int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_bf16(svwhilelt_b16(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<c10::BFloat16> loadu(
      const void* ptr,
      int64_t count = size()) {
    if (count == size()) {
      return vld1q_bf16(reinterpret_cast<const bfloat16_t*>(ptr));
    }
    svbool_t pg = svwhilelt_b16(0ull, count);
    return svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_bf16(reinterpret_cast<bfloat16_t*>(ptr), values);
      return;
    } else {
      svbool_t pg = svwhilelt_b16(0ull, count);
      svst1_bf16(pg, reinterpret_cast<bfloat16_t*>(ptr), valuesAsSve());
    }
  }
  inline c10::BFloat16 operator[](int idx) const {
    return values[idx];
  }
  inline c10::BFloat16 operator[](int idx) {
    return values[idx];
  }
  bool has_inf_nan() const;
#define DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(name)    \
  Vectorized name() const {                                     \
    return map_with_vec_float_method(&Vectorized<float>::name); \
  }

#define DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(name) \
  Vectorized name(const Vectorized& other) const {               \
    return map2_bitmask_with_vec_float_method(                   \
        other, &Vectorized<float>::name);                        \
  }

  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(abs)
  Vectorized frac() const;
  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(isnan)
  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(neg)
  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(trunc)
  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(sqrt)
  DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD(reciprocal)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator==)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator!=)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator<)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator<=)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator>)
  DEFINE_BINARY_COMPARISON_OPERATOR_VIA_FLOAT_METHOD(operator>=)

#undef DEFINE_UNARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD
#undef DEFINE_BINARY_ELEMENTWISE_FUNC_VIA_FLOAT_METHOD

  Vectorized eq(const Vectorized& other) const;
  Vectorized ne(const Vectorized& other) const;
  Vectorized gt(const Vectorized& other) const;
  Vectorized ge(const Vectorized& other) const;
  Vectorized lt(const Vectorized& other) const;
  Vectorized le(const Vectorized& other) const;

  template <typename step_t>
  static Vectorized<BFloat16> arange(
      BFloat16 base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    bfloat16x8_t result;
    result[0] = base;
    result[1] = base + step;
    result[2] = base + step * 2;
    result[3] = base + step * 3;
    result[4] = base + step * 4;
    result[5] = base + step * 5;
    result[6] = base + step * 6;
    result[7] = base + step * 7;
    return result;
  }

}; // Vectorized<c10::BFloat16>

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<c10::BFloat16>& a) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  bfloat16x8_t x = a;
  float32x4_t x1 =
      Vectorized<c10::BFloat16>::convert_f32_bf16(vget_low_bf16(x));
  float32x4_t x2 =
      Vectorized<c10::BFloat16>::convert_f32_bf16(vget_high_bf16(x));
  return {Vectorized<float>(x1), Vectorized<float>(x2)};
}
inline Vectorized<c10::BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  bfloat16x4_t x1 = Vectorized<c10::BFloat16>::convert_bf16_f32(a);
  bfloat16x4_t x2 = Vectorized<c10::BFloat16>::convert_bf16_f32(b);
  return Vectorized<c10::BFloat16>(vcombine_bf16(x1, x2));
}

inline void load_fp32_from_bf16(const BFloat16* data, Vectorized<float>& out) {
  const bfloat16_t* dataPtr = reinterpret_cast<const bfloat16_t*>(data);
  bfloat16x4_t lowbf16 = vld1_bf16(dataPtr);
  out = vcvt_f32_bf16(lowbf16);
}

inline void load_fp32_from_bf16(
    const BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  Vectorized<BFloat16> bf16_vec = Vectorized<BFloat16>::loadu(data);
  auto floats = convert_bfloat16_float(bf16_vec);
  out1 = std::get<0>(floats);
  out2 = std::get<1>(floats);
}

bool inline Vectorized<c10::BFloat16>::has_inf_nan() const {
  auto [v1, v2] = convert_bfloat16_float(values);
  return v1.has_inf_nan() || v2.has_inf_nan();
}

template <typename Op>
Vectorized<c10::BFloat16> binary_operator_via_float(
    Op op,
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return convert_float_bfloat16(
      op(a_float_low, b_float_low), op(a_float_high, b_float_high));
}

template <>
Vectorized<c10::BFloat16> inline operator+(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::plus<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator-(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::minus<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator*(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::multiplies<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator/(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::divides<Vectorized<float>>(), a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::frac() const {
  return *this - this->trunc();
}

template <>
Vectorized<c10::BFloat16> inline maximum(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&maximum),
      a,
      b);
}

template <>
Vectorized<c10::BFloat16> inline minimum(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&minimum),
      a,
      b);
}

template <>
Vectorized<c10::BFloat16> inline clamp(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& min,
    const Vectorized<c10::BFloat16>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<c10::BFloat16> inline clamp_max(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& max) {
  return minimum(max, a);
}

template <>
Vectorized<c10::BFloat16> inline clamp_min(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& min) {
  return maximum(min, a);
}

template <>
Vectorized<c10::BFloat16> inline operator&(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return Vectorized<c10::BFloat16>(vreinterpretq_bf16_u16(
      vandq_u16(vreinterpretq_u16_bf16(a), vreinterpretq_u16_bf16(b))));
}

template <>
Vectorized<c10::BFloat16> inline operator|(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return Vectorized<c10::BFloat16>(vreinterpretq_bf16_u16(
      vorrq_u16(vreinterpretq_u16_bf16(a), vreinterpretq_u16_bf16(b))));
}

template <>
Vectorized<c10::BFloat16> inline operator^(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return Vectorized<c10::BFloat16>(vreinterpretq_bf16_u16(
      veorq_u16(vreinterpretq_u16_bf16(a), vreinterpretq_u16_bf16(b))));
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::eq(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this == other) & Vectorized<c10::BFloat16>(1);
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::ne(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this != other) & Vectorized<c10::BFloat16>(1);
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::gt(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this > other) & Vectorized<c10::BFloat16>(1);
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::ge(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this >= other) & Vectorized<c10::BFloat16>(1);
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::lt(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this < other) & Vectorized<c10::BFloat16>(1);
}

inline Vectorized<c10::BFloat16> Vectorized<c10::BFloat16>::le(
    const Vectorized<c10::BFloat16>& other) const {
  return (*this <= other) & Vectorized<c10::BFloat16>(1);
}

template <>
Vectorized<c10::BFloat16> inline fmadd(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b,
    const Vectorized<c10::BFloat16>& c) {
  const auto [lowA, highA] = convert_bfloat16_float(a);
  const auto [lowB, highB] = convert_bfloat16_float(b);
  const auto [lowC, highC] = convert_bfloat16_float(c);
  Vectorized<float> resultLow = fmadd(lowA, lowB, lowC);
  Vectorized<float> resultHigh = fmadd(highA, highB, highC);
  return convert_float_bfloat16(resultLow, resultHigh);
}

template <>
Vectorized<c10::BFloat16> inline fnmadd(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b,
    const Vectorized<c10::BFloat16>& c) {
  const auto [lowA, highA] = convert_bfloat16_float(a);
  const auto [lowB, highB] = convert_bfloat16_float(b);
  const auto [lowC, highC] = convert_bfloat16_float(c);
  Vectorized<float> resultLow = fnmadd(lowA, lowB, lowC);
  Vectorized<float> resultHigh = fnmadd(highA, highB, highC);
  return convert_float_bfloat16(resultLow, resultHigh);
}

template <>
Vectorized<c10::BFloat16> inline fmsub(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b,
    const Vectorized<c10::BFloat16>& c) {
  const auto [lowA, highA] = convert_bfloat16_float(a);
  const auto [lowB, highB] = convert_bfloat16_float(b);
  const auto [lowC, highC] = convert_bfloat16_float(c);
  Vectorized<float> resultLow = fmsub(lowA, lowB, lowC);
  Vectorized<float> resultHigh = fmsub(highA, highB, highC);
  return convert_float_bfloat16(resultLow, resultHigh);
}

template <>
Vectorized<c10::BFloat16> inline fnmsub(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b,
    const Vectorized<c10::BFloat16>& c) {
  const auto [lowA, highA] = convert_bfloat16_float(a);
  const auto [lowB, highB] = convert_bfloat16_float(b);
  const auto [lowC, highC] = convert_bfloat16_float(c);
  Vectorized<float> resultLow = fnmsub(lowA, lowB, lowC);
  Vectorized<float> resultHigh = fnmsub(highA, highB, highC);
  return convert_float_bfloat16(resultLow, resultHigh);
}

#else //

CONVERT_NON_VECTORIZED_INIT(BFloat16, bfloat16)

LOAD_FP32_NON_VECTORIZED_INIT(BFloat16, bf16)

#endif // !defined(C10_MOBILE) && defined(__aarch64__)

} // namespace CPU_CAPABILITY
} // namespace at::vec
