#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>

#if defined(__aarch64__) && defined(AT_BUILD_ARM_VEC256_WITH_SLEEF)
#include <sleef.h>
#endif

// Sleef offers vectorized versions of some transcedentals
// such as sin, cos, tan etc..
// However for now opting for STL, since we are not building
// with Sleef for mobile yet.

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
#if defined(__aarch64__)

#ifdef __BIG_ENDIAN__
#error "Big endian is not supported."
#endif

#if defined(AT_BUILD_ARM_VEC256_WITH_SLEEF)
#define USE_SLEEF(sleef_code, non_sleef_code) sleef_code
#else
#define USE_SLEEF(sleef_code, non_sleef_code) non_sleef_code
#endif

template <int index, bool mask_val>
struct BlendRegs {
  static float32x4_t impl(
      const float32x4_t& a,
      const float32x4_t& b,
      float32x4_t& res);
};

template <int index>
struct BlendRegs<index, true> {
  static float32x4_t impl(
      const float32x4_t& a,
      const float32x4_t& b,
      float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(b, index), res, index);
  }
};

template <int index>
struct BlendRegs<index, false> {
  static float32x4_t impl(
      const float32x4_t& a,
      const float32x4_t& b,
      float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(a, index), res, index);
  }
};

template <>
struct is_vec_specialized_for<float> : std::bool_constant<true> {};

template <>
class Vectorized<float> {
 private:
  float32x4_t values;

 public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  Vectorized(float32x4_t v) : values(v) {}
  Vectorized(float val) : values{vdupq_n_f32(val)} {}
  Vectorized(float val0, float val1, float val2, float val3)
      : values{val0, val1, val2, val3} {}
  Vectorized(float (&arr)[4]) : Vectorized(arr[0], arr[1], arr[2], arr[3]) {}
  operator float32x4_t() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    Vectorized<float> vec;
    vec.values = BlendRegs < 0,
    (mask & 0x01) != 0 > ::impl(a.values, b.values, vec.values);
    vec.values = BlendRegs < 1,
    (mask & 0x02) != 0 > ::impl(a.values, b.values, vec.values);
    vec.values = BlendRegs < 2,
    (mask & 0x04) != 0 > ::impl(a.values, b.values, vec.values);
    vec.values = BlendRegs < 3,
    (mask & 0x08) != 0 > ::impl(a.values, b.values, vec.values);
    return vec;
  }
  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask) {
    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    Vectorized<float> vec(mask.values);
    vec.values =
        vbslq_f32(vreinterpretq_u32_f32(vec.values), b.values, a.values);
    return vec;
  }
  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<float> base_vec(base);
    const Vectorized<float> step_vec(step);
    const Vectorized<float> step_sizes(0, 1, 2, 3);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1: {
        Vectorized<float> vec;
        static uint32x4_t mask_low = {0xFFFFFFFF, 0x0, 0x0, 0x0};
        vec.values = vreinterpretq_f32_u32(mask_low);
        vec.values =
            vbslq_f32(vreinterpretq_u32_f32(vec.values), b.values, a.values);
        return vec;
      }
      case 2: {
        Vectorized<float> vec;
        static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
        vec.values = vreinterpretq_f32_u32(mask_low);
        vec.values =
            vbslq_f32(vreinterpretq_u32_f32(vec.values), b.values, a.values);
        return vec;
      }
      case 3: {
        Vectorized<float> vec;
        static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
        vec.values = vreinterpretq_f32_u32(mask_low);
        vec.values =
            vbslq_f32(vreinterpretq_u32_f32(vec.values), b.values, a.values);
        return vec;
      }
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f32(reinterpret_cast<const float*>(ptr));
    } else {
      __at_align__ float tmp_values[size()];
      for (const auto i : c10::irange(size())) {
        tmp_values[i] = 0.0;
      }
      std::memcpy(
          tmp_values,
          reinterpret_cast<const float*>(ptr),
          count * sizeof(float));
      return vld1q_f32(reinterpret_cast<const float*>(tmp_values));
    }
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f32(reinterpret_cast<float*>(ptr), values);
    } else {
      float tmp_values[size()];
      vst1q_f32(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  // Very slow implementation of indexing.
  // Only required because vec256_qint refers to this.
  // Once we specialize that implementation for ARM
  // this should be removed. TODO (kimishpatel)
  float operator[](int idx) const {
    __at_align__ float tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  float operator[](int idx) {
    __at_align__ float tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  // For boolean version where we want to if any 1/all zero
  // etc. can be done faster in a different way.
  int zero_mask() const {
    __at_align__ float tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0.f) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<float> isnan() const {
    return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(values, values)));
  }
  bool has_inf_nan() const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i]) || _isinf(tmp[i])) {
        return true;
      }
    }
    return false;
  }
  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> map2(
      const Vectorized<float>& second,
      float (*const f)(float, float)) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_second[size()];
    store(tmp);
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i], tmp_second[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    return Vectorized<float>(vabsq_f32(values));
  }
  Vectorized<float> angle() const {
    auto zero = Vectorized<float>(0);
    auto pi = Vectorized<float>(c10::pi<float>);
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<float> real() const {
    return *this;
  }
  Vectorized<float> imag() const {
    return Vectorized<float>(0.f);
  }
  Vectorized<float> conj() const {
    return *this;
  }
#define DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(      \
    name, sleef_name)                                                        \
  Vectorized<float> name() const {                                           \
    return USE_SLEEF(Vectorized<float>(sleef_name(values)), map(std::name)); \
  }

#define DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(name)      \
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME( \
      name, Sleef_##name##f4_u10)

  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(acos)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(acosh)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(asin)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(asinh)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(atan)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(atanh)

#define DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME( \
    name, sleef_name)                                                    \
  Vectorized<float> name(const Vectorized<float>& arg) const {           \
    return USE_SLEEF(                                                    \
        Vectorized<float>(sleef_name(values, arg.values)),               \
        map2(arg, std::name));                                           \
  }

#define DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC(name)      \
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME( \
      name, Sleef_##name##f4_u10)

  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC(atan2)
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(
      copysign,
      Sleef_copysignf4)
  Vectorized<float> erf() const;
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(
      erfc,
      Sleef_erfcf4_u15)
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(exp)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(exp2)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(expm1)
  Vectorized<float> exp_u20() const {
    return exp();
  }
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(
      fmod,
      Sleef_fmodf4)
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(
      hypot,
      Sleef_hypotf4_u05)
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> igamma(const Vectorized<float>& x) const {
    return map2(x, calc_igamma);
  }
  Vectorized<float> igammac(const Vectorized<float>& x) const {
    return map2(x, calc_igammac);
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(log)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(log10)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(log1p)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(log2)
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC_WITH_SLEEF_NAME(
      nextafter,
      Sleef_nextafterf4)
  Vectorized<float> frac() const;
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(sin)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(sinh)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(cos)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(cosh)
  Vectorized<float> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<float> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<float> neg() const {
    return Vectorized<float>(vnegq_f32(values));
  }
  Vectorized<float> round() const {
    // We do not use std::round because we would like to round midway numbers to
    // the nearest even integer.
    return map(at::native::round_impl);
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(tan)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(tanh)
  Vectorized<float> trunc() const {
    return Vectorized<float>(vrndq_f32(values));
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(lgamma)
  Vectorized<float> sqrt() const {
    return Vectorized<float>(vsqrtq_f32(values));
  }
  Vectorized<float> reciprocal() const {
    return Vectorized<float>(vdivq_f32(vdupq_n_f32(1.0f), values));
  }
  Vectorized<float> rsqrt() const {
    return this->sqrt().reciprocal();
  }
  DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC(pow)
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    return Vectorized<float>(
        vreinterpretq_f32_u32(vceqq_f32(values, other.values)));
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    float32x4_t r0 =
        vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(values, other.values)));
    return Vectorized<float>(r0);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    return Vectorized<float>(
        vreinterpretq_f32_u32(vcltq_f32(values, other.values)));
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    return Vectorized<float>(
        vreinterpretq_f32_u32(vcleq_f32(values, other.values)));
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    return Vectorized<float>(
        vreinterpretq_f32_u32(vcgtq_f32(values, other.values)));
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    return Vectorized<float>(
        vreinterpretq_f32_u32(vcgeq_f32(values, other.values)));
  }

  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vaddq_f32(a, b));
}

template <>
Vectorized<float> inline operator-(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vsubq_f32(a, b));
}

template <>
Vectorized<float> inline operator*(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vmulq_f32(a, b));
}

template <>
Vectorized<float> inline operator/(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vdivq_f32(a, b));
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

template <>
Vectorized<float> inline maximum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vmaxq_f32(a, b));
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vminq_f32(a, b));
}

template <>
Vectorized<float> inline clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min,
    const Vectorized<float>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<float> inline clamp_max(
    const Vectorized<float>& a,
    const Vectorized<float>& max) {
  return minimum(max, a);
}

template <>
Vectorized<float> inline clamp_min(
    const Vectorized<float>& a,
    const Vectorized<float>& min) {
  return maximum(min, a);
}

template <>
Vectorized<float> inline operator&(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vreinterpretq_f32_u32(
      vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))));
}

template <>
Vectorized<float> inline operator|(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vreinterpretq_f32_u32(
      vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))));
}

template <>
Vectorized<float> inline operator^(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return Vectorized<float>(vreinterpretq_f32_u32(
      veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))));
}

inline Vectorized<float> Vectorized<float>::eq(
    const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(
    const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(
    const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(
    const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(
    const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(
    const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, int32_t* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    vst1q_s32(dst + i, vcvtq_s32_f32(vld1q_f32(src + i)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<int32_t>(src[i]);
  }
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  int64_t i;
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (i = 0; i <= (n - Vectorized<float>::size());
       i += Vectorized<float>::size()) {
    vst1q_f32(dst + i, vcvtq_f32_s32(vld1q_s32(src + i)));
  }
#ifndef __msvc_cl__
#pragma unroll
#endif
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
Vectorized<float> inline fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return Vectorized<float>(vfmaq_f32(c, a, b));
}

template <>
Vectorized<float> inline fmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return Vectorized<float>(vnegq_f32(vfmsq_f32(c, a, b)));
}

inline Vectorized<float> Vectorized<float>::erf() const {
  // constants
  const Vectorized<float> neg_zero_vec(-0.f);
  const Vectorized<float> one_vec(1.0f);
  const Vectorized<float> p(0.3275911f);
  const Vectorized<float> p1(0.254829592f);
  const Vectorized<float> p2(-0.284496736f);
  const Vectorized<float> p3(1.421413741f);
  const Vectorized<float> p4(-1.453152027f);
  const Vectorized<float> p5(1.061405429f);
  // sign(x)
  auto sign_mask = neg_zero_vec & *this;
  auto abs_vec = this->abs();
  // t = 1 / (p * abs(x) + 1)
  auto tmp0 = fmadd(p, abs_vec, one_vec);
  auto t = one_vec / tmp0;
  // r = p5 * t ^ 4 + p4 * t ^ 3 + p3 * t ^ 2 + p2 * t + p1
  auto tmp1 = fmadd(p5, t, p4);
  auto tmp2 = fmadd(tmp1, t, p3);
  auto tmp3 = fmadd(tmp2, t, p2);
  auto r = fmadd(tmp3, t, p1);
  // - exp(- x * x)
  auto pow_2 = (*this) * (*this);
  auto neg_pow_2 = pow_2 ^ neg_zero_vec;
  auto tmp4 = neg_pow_2.map(
      std::exp); // This can be swapped for a faster implementation of exp.
  auto tmp5 = tmp4 ^ neg_zero_vec;
  // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
  auto tmp6 = t * tmp5;
  auto tmp7 = fmadd(tmp6, r, one_vec);
  return tmp7 ^ sign_mask;
}
#undef DEFINE_SLEEF_COMPATIBLE_BINARY_ELEMENTWISE_FUNC
#undef DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC
#endif /* defined(aarch64) */

} // namespace CPU_CAPABILITY
} // namespace at::vec
