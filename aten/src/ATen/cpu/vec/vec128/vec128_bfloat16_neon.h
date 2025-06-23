#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]
#include <ATen/cpu/vec/vec128/vec128_float_neon.h>
#include <ATen/cpu/vec/vec128/vec128_reduced_precision_common_neon.h>
#include <ATen/cpu/vec/vec_base.h>
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

// Unlike the float16_t family of types, bfloat16_t is not available
// when we're not targeting bfloat16 hardware support on some
// platforms (but not Mac, so we have to be careful not to shadow the
// definitions in case they are actually there!). (See
// https://godbolt.org/z/orv6e94n4 ) So, we need to handle it as
// uint16_t in that case.
#define IMPLEMENT_AT_BF16_SHIM(vec_suffix)                               \
  inline at_bfloat16x4_t at_vget_low_bf16(at_bfloat16x8_t a) {           \
    return vget_low_##vec_suffix(a);                                     \
  }                                                                      \
                                                                         \
  inline at_bfloat16x4_t at_vget_high_bf16(at_bfloat16x8_t a) {          \
    return vget_high_##vec_suffix(a);                                    \
  }                                                                      \
                                                                         \
  inline at_bfloat16x8_t at_vcombine_bf16(                               \
      at_bfloat16x4_t low, at_bfloat16x4_t high) {                       \
    return vcombine_##vec_suffix(low, high);                             \
  }                                                                      \
                                                                         \
  inline at_bfloat16x8_t at_vdupq_n_bf16(at_bfloat16_t value) {          \
    return vdupq_n_##vec_suffix(value);                                  \
  }                                                                      \
                                                                         \
  inline at_bfloat16x8_t at_vld1q_bf16(const at_bfloat16_t* ptr) {       \
    return vld1q_##vec_suffix(ptr);                                      \
  }                                                                      \
                                                                         \
  inline void at_vst1q_bf16(at_bfloat16_t* ptr, at_bfloat16x8_t value) { \
    vst1q_##vec_suffix(ptr, value);                                      \
  }                                                                      \
                                                                         \
  template <typename T>                                                  \
  inline at_bfloat16x8_t at_vreinterpretq_bf16_u16(T val) {              \
    if constexpr (std::is_same_v<at_bfloat16x8_t, uint16x8_t>) {         \
      return val;                                                        \
    } else {                                                             \
      return vreinterpretq_bf16_u16(val);                                \
    }                                                                    \
  }                                                                      \
  template <typename T>                                                  \
  inline at_bfloat16x4_t at_vreinterpret_bf16_u16(T val) {               \
    if constexpr (std::is_same_v<at_bfloat16x4_t, uint16x4_t>) {         \
      return val;                                                        \
    } else {                                                             \
      return vreinterpret_bf16_u16(val);                                 \
    }                                                                    \
  }                                                                      \
  template <typename T>                                                  \
  inline uint16x8_t at_vreinterpretq_u16_bf16(T val) {                   \
    if constexpr (std::is_same_v<at_bfloat16x8_t, uint16x8_t>) {         \
      return val;                                                        \
    } else {                                                             \
      return vreinterpretq_u16_bf16(val);                                \
    }                                                                    \
  }                                                                      \
  template <typename T>                                                  \
  inline uint16x4_t at_vreinterpret_u16_bf16(T val) {                    \
    if constexpr (std::is_same_v<at_bfloat16x4_t, uint16x4_t>) {         \
      return val;                                                        \
    } else {                                                             \
      return vreinterpret_u16_bf16(val);                                 \
    }                                                                    \
  }

#ifdef __ARM_FEATURE_BF16
using at_bfloat16x8_t = bfloat16x8_t;
using at_bfloat16x4_t = bfloat16x4_t;
using at_bfloat16_t = bfloat16_t;
IMPLEMENT_AT_BF16_SHIM(bf16)
#define at_vsetq_lane_bf16 vsetq_lane_bf16
#define at_vgetq_lane_bf16 vgetq_lane_bf16
#else
using at_bfloat16x8_t = uint16x8_t;
using at_bfloat16x4_t = uint16x4_t;
using at_bfloat16_t = uint16_t;
IMPLEMENT_AT_BF16_SHIM(u16)
#define at_vsetq_lane_bf16 vsetq_lane_u16
#define at_vgetq_lane_bf16 vgetq_lane_u16
#endif // __ARM_FEATURE_BF16

template <int index, bool mask_val>
struct BlendBFloat16Regs {
  static at_bfloat16x8_t impl(
      const at_bfloat16x8_t& a,
      const at_bfloat16x8_t& b,
      at_bfloat16x8_t& res);
};

template <int index>
struct BlendBFloat16Regs<index, true> {
  static at_bfloat16x8_t impl(
      const at_bfloat16x8_t& a,
      const at_bfloat16x8_t& b,
      at_bfloat16x8_t& res) {
    return at_vsetq_lane_bf16(at_vgetq_lane_bf16(b, index), res, index);
  }
};

template <int index>
struct BlendBFloat16Regs<index, false> {
  static at_bfloat16x8_t impl(
      const at_bfloat16x8_t& a,
      const at_bfloat16x8_t& b,
      at_bfloat16x8_t& res) {
    return at_vsetq_lane_bf16(at_vgetq_lane_bf16(a, index), res, index);
  }
};

template <>
struct is_vec_specialized_for<c10::BFloat16> : std::bool_constant<true> {};

template <>
class Vectorized<c10::BFloat16> : public Vectorized16<
                                      at_bfloat16x8_t,
                                      c10::BFloat16,
                                      BlendBFloat16Regs,
                                      Vectorized<c10::BFloat16>> {
  using Base = Vectorized16<
      at_bfloat16x8_t,
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
    __at_align__ c10::BFloat16 tmp_first[size()];
    __at_align__ c10::BFloat16 tmp_second[size()];
    store(tmp_first); // store this to tmp_first
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]);
    }
    return loadu(tmp_first);
  }

  static float32x4_t convert_f32_bf16(at_bfloat16x4_t bf16) {
#ifdef __ARM_FEATURE_BF16
    return vcvt_f32_bf16(bf16);
#else
    int32x4_t shift = vdupq_n_s32(16);
    return vreinterpretq_f32_u32(vshlq_u32(vmovl_u16(bf16), shift));
#endif // __ARM_FEATURE_BF16
  }

  static at_bfloat16x4_t convert_bf16_f32(const Vectorized<float>& f32) {
#ifdef __ARM_FEATURE_BF16
    return vcvt_bf16_f32(f32);
#else
    static_assert(std::is_same_v<uint16x4_t, at_bfloat16x4_t>);
    uint32x4_t as_uint32 = vreinterpretq_u32_f32(f32);
    uint32x4_t rounding_bias = vaddq_u32(
        vandq_u32(vshrq_n_u32(as_uint32, 16), vdupq_n_u32(1)),
        vdupq_n_u32(0x7FFF));
    at_bfloat16x4_t rounded =
        vshrn_n_u32(vaddq_u32(as_uint32, rounding_bias), 16);
    const auto bf16_nan = vdup_n_u16(0x7FC0);
    return vbsl_u16(
        vmovn_u32(vreinterpretq_u32_f32(f32.isnan())), bf16_nan, rounded);
#endif // __ARM_FEATURE_BF16
  }

  Vectorized<c10::BFloat16> map_with_vec_float_method(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    float32x4_t v00 = convert_f32_bf16(at_vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(at_vget_high_bf16(values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)();
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)();
    at_bfloat16x4_t r00 = convert_bf16_f32(mv0);
    at_bfloat16x4_t r01 = convert_bf16_f32(mv1);
    return Vectorized<c10::BFloat16>(at_vcombine_bf16(r00, r01));
  }

  Vectorized<c10::BFloat16> map2_with_vec_float_method(
      const Vectorized<c10::BFloat16>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = convert_f32_bf16(at_vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(at_vget_high_bf16(values));
    float32x4_t second_v00 = convert_f32_bf16(at_vget_low_bf16(second.values));
    float32x4_t second_v01 = convert_f32_bf16(at_vget_high_bf16(second.values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)(second_v00);
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)(second_v01);
    at_bfloat16x4_t r00 = convert_bf16_f32(mv0);
    at_bfloat16x4_t r01 = convert_bf16_f32(mv1);
    return Vectorized<c10::BFloat16>(at_vcombine_bf16(r00, r01));
  }

  Vectorized<c10::BFloat16> map2_bitmask_with_vec_float_method(
      const Vectorized<c10::BFloat16>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    float32x4_t v00 = convert_f32_bf16(at_vget_low_bf16(values));
    float32x4_t v01 = convert_f32_bf16(at_vget_high_bf16(values));
    float32x4_t second_v00 = convert_f32_bf16(at_vget_low_bf16(second.values));
    float32x4_t second_v01 = convert_f32_bf16(at_vget_high_bf16(second.values));
    Vectorized<float> mv0 = (Vectorized<float>(v00).*m)(second_v00);
    Vectorized<float> mv1 = (Vectorized<float>(v01).*m)(second_v01);
    // Assume the operator returns a bitmask, not "real" floats, and
    // just narrow the bits. All-ones is a NaN and will get mangled by
    // conversion!
    at_bfloat16x4_t r00 =
        at_vreinterpret_bf16_u16(vmovn_u32(vreinterpretq_u32_f32(mv0)));
    at_bfloat16x4_t r01 =
        at_vreinterpret_bf16_u16(vmovn_u32(vreinterpretq_u32_f32(mv1)));
    return Vectorized<c10::BFloat16>(at_vcombine_bf16(r00, r01));
  }

 public:
  using Vectorized16::Vectorized16;

  Vectorized() = default;

  Vectorized(c10::BFloat16 val)
      : Vectorized16(at_vdupq_n_bf16(c10::bit_cast<at_bfloat16_t>(val.x))) {}
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
      : Vectorized16(at_bfloat16x8_t{
            c10::bit_cast<at_bfloat16_t>(val0.x),
            c10::bit_cast<at_bfloat16_t>(val1.x),
            c10::bit_cast<at_bfloat16_t>(val2.x),
            c10::bit_cast<at_bfloat16_t>(val3.x),
            c10::bit_cast<at_bfloat16_t>(val4.x),
            c10::bit_cast<at_bfloat16_t>(val5.x),
            c10::bit_cast<at_bfloat16_t>(val6.x),
            c10::bit_cast<at_bfloat16_t>(val7.x)}) {}

  static Vectorized<c10::BFloat16> blendv(
      const Vectorized<c10::BFloat16>& a,
      const Vectorized<c10::BFloat16>& b,
      const Vectorized<c10::BFloat16>& mask) {
    // NOTE: blendv has the same problems as it does for Half; see comments in
    // vec128_half_neon.h.
    Vectorized<c10::BFloat16> vec(mask.values);
    vec.values = at_vreinterpretq_bf16_u16(vbslq_u16(
        at_vreinterpretq_u16_bf16(vec.values),
        at_vreinterpretq_u16_bf16(b.values),
        at_vreinterpretq_u16_bf16(a.values)));
    return vec;
  }
  static Vectorized<c10::BFloat16> set(
      const Vectorized<c10::BFloat16>& a,
      const Vectorized<c10::BFloat16>& b,
      int64_t count = size()) {
    uint16_t pre_mask[size()] = {0};
    for (int i = 0; i < count; i++) {
      pre_mask[i] = 0xFFFF;
    }
    uint16x8_t mask = vld1q_u16(pre_mask);

    Vectorized<c10::BFloat16> vec(at_vreinterpretq_bf16_u16(vbslq_u16(
        mask,
        at_vreinterpretq_u16_bf16(b.values),
        at_vreinterpretq_u16_bf16(a.values))));

    return vec;
  }
  static Vectorized<c10::BFloat16> loadu(
      const void* ptr,
      int64_t count = size()) {
    if (count == size()) {
      return at_vld1q_bf16(reinterpret_cast<const at_bfloat16_t*>(ptr));
    }
    __at_align__ at_bfloat16_t tmp_values[size()];
    std::memset(tmp_values, 0, sizeof(tmp_values));
    std::memcpy(
        tmp_values,
        reinterpret_cast<const at_bfloat16_t*>(ptr),
        count * sizeof(at_bfloat16_t));
    return at_vld1q_bf16(reinterpret_cast<const at_bfloat16_t*>(tmp_values));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      at_vst1q_bf16(reinterpret_cast<at_bfloat16_t*>(ptr), values);
      return;
    } else {
      at_bfloat16_t tmp_values[size()];
      at_vst1q_bf16(reinterpret_cast<at_bfloat16_t*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(at_bfloat16_t));
    }
  }
  Vectorized<c10::BFloat16> isnan() const {
    // NOTE: we could make this faster by doing vectorized checks of
    // exponent/payload bits.
    __at_align__ c10::BFloat16 tmp[size()];
    __at_align__ c10::BFloat16 res[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i])) {
        std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(c10::BFloat16));
      } else {
        std::memset(static_cast<void*>(&res[i]), 0, sizeof(c10::BFloat16));
      }
    }
    return loadu(res);
  }
  bool has_inf_nan() const {
    __at_align__ c10::BFloat16 tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }
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
}; // Vectorized<c10::BFloat16>

inline std::tuple<Vectorized<float>, Vectorized<float>> convert_bfloat16_float(
    const Vectorized<c10::BFloat16>& a) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  at_bfloat16x8_t x = a;
  float32x4_t x1 =
      Vectorized<c10::BFloat16>::convert_f32_bf16(at_vget_low_bf16(x));
  float32x4_t x2 =
      Vectorized<c10::BFloat16>::convert_f32_bf16(at_vget_high_bf16(x));
  return {Vectorized<float>(x1), Vectorized<float>(x2)};
}
inline Vectorized<c10::BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  at_bfloat16x4_t x1 = Vectorized<c10::BFloat16>::convert_bf16_f32(a);
  at_bfloat16x4_t x2 = Vectorized<c10::BFloat16>::convert_bf16_f32(b);
  return Vectorized<c10::BFloat16>(at_vcombine_bf16(x1, x2));
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
  return Vectorized<c10::BFloat16>(at_vreinterpretq_bf16_u16(
      vandq_u16(at_vreinterpretq_u16_bf16(a), at_vreinterpretq_u16_bf16(b))));
}

template <>
Vectorized<c10::BFloat16> inline operator|(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return Vectorized<c10::BFloat16>(at_vreinterpretq_bf16_u16(
      vorrq_u16(at_vreinterpretq_u16_bf16(a), at_vreinterpretq_u16_bf16(b))));
}

template <>
Vectorized<c10::BFloat16> inline operator^(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return Vectorized<c10::BFloat16>(at_vreinterpretq_bf16_u16(
      veorq_u16(at_vreinterpretq_u16_bf16(a), at_vreinterpretq_u16_bf16(b))));
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
  // NOTE [BF16 FMA]: There isn't an FMA that accumulates into BF16!  Also,
  // vbfmlalbq_f32 and vbfmlaltq_f32 take the even and odd-numbered
  // elements, not the bottom and top half, so they don't seem
  // particularly useful here. Ideally we would include dot product in
  // the Vectorized interface...
  return a * b + c;
}

template <>
Vectorized<c10::BFloat16> inline fmsub(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b,
    const Vectorized<c10::BFloat16>& c) {
  // See NOTE [BF16 FMA] above.
  return a * b - c;
}

#endif // !defined(C10_MOBILE) && defined(__aarch64__)

} // namespace CPU_CAPABILITY
} // namespace at::vec
