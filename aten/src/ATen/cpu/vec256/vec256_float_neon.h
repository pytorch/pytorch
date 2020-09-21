#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
// Sleef offers vectorized versions of some transcedentals
// such as sin, cos, tan etc..
// However for now opting for STL, since we are not building
// with Sleef for mobile yet.

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

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

template<int index, bool mask_val>
struct BlendRegs {
  static float32x4_t impl(
    const float32x4_t& a, const float32x4_t& b, float32x4_t& res);
};

template<int index>
struct BlendRegs<index, true>{
  static float32x4_t impl(
      const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(b, index), res, index);
  }
};

template<int index>
struct BlendRegs<index, false>{
  static float32x4_t impl(
      const float32x4_t& a, const float32x4_t& b, float32x4_t& res) {
    return vsetq_lane_f32(vgetq_lane_f32(a, index), res, index);
  }
};

template <> class Vec256<float> {
private:
  float32x4x2_t values;
public:
  using value_type = float;
  static constexpr int size() {
    return 8;
  }
  Vec256() {}
  Vec256(float32x4x2_t v) : values(v) {}
  Vec256(float val) : values{vdupq_n_f32(val), vdupq_n_f32(val) } {}
  Vec256(float val0, float val1, float val2, float val3,
         float val4, float val5, float val6, float val7) :
         values{val0, val1, val2, val3, val4, val5, val6, val7} {}
  Vec256(float32x4_t val0, float32x4_t val1) : values{val0, val1} {}
  operator float32x4x2_t() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<float> blend(const Vec256<float>& a, const Vec256<float>& b) {
    Vec256<float> vec;
    // 0.
    vec.values.val[0] =
      BlendRegs<0, (mask & 0x01)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] =
      BlendRegs<1, (mask & 0x02)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] =
      BlendRegs<2, (mask & 0x04)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] =
      BlendRegs<3, (mask & 0x08)!=0>::impl(
          a.values.val[0], b.values.val[0], vec.values.val[0]);
    // 1.
    vec.values.val[1] =
      BlendRegs<0, (mask & 0x10)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] =
      BlendRegs<1, (mask & 0x20)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] =
      BlendRegs<2, (mask & 0x40)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] =
      BlendRegs<3, (mask & 0x80)!=0>::impl(
          a.values.val[1], b.values.val[1], vec.values.val[1]);
    return vec;
  }
  static Vec256<float> blendv(const Vec256<float>& a, const Vec256<float>& b,
                              const Vec256<float>& mask) {
    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    Vec256<float> vec(mask.values);
    vec.values.val[0] = vbslq_f32(
        vreinterpretq_u32_f32(vec.values.val[0]),
        b.values.val[0],
        a.values.val[0]);
    vec.values.val[1] = vbslq_f32(
        vreinterpretq_u32_f32(vec.values.val[1]),
        b.values.val[1],
        a.values.val[1]);
    return vec;
  }
  template<typename step_t>
  static Vec256<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    const Vec256<float> base_vec(base);
    const Vec256<float> step_vec(step);
    const Vec256<float> step_sizes(0, 1, 2, 3, 4, 5, 6, 7);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vec256<float> set(const Vec256<float>& a, const Vec256<float>& b,
                           int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        {
          Vec256<float> vec;
          static uint32x4_t mask_low = {0xFFFFFFFF, 0x0, 0x0, 0x0};
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          vec.values.val[1] = a.values.val[1];
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          return vec;
        }
      case 2:
        {
          Vec256<float> vec;
          static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          vec.values.val[1] = a.values.val[1];
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          return vec;
        }
      case 3:
        {
          Vec256<float> vec;
          static uint32x4_t mask_low = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
          vec.values.val[0] = vreinterpretq_f32_u32(mask_low);
          vec.values.val[1] = a.values.val[1];
          vec.values.val[0] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[0]),
              b.values.val[0],
              a.values.val[0]);
          return vec;
        }
      case 4:
        return Vec256<float>(b.values.val[0], a.values.val[1]);
      case 5:
        {
          Vec256<float> vec;
          static uint32x4_t mask_high = {0xFFFFFFFF, 0x0, 0x0, 0x0};
          vec.values.val[0] = b.values.val[0];
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          return vec;
        }
      case 6:
        {
          Vec256<float> vec;
          static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0x0, 0x0};
          vec.values.val[0] = b.values.val[0];
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          return vec;
        }
      case 7:
        {
          Vec256<float> vec;
          static uint32x4_t mask_high = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0};
          vec.values.val[0] = b.values.val[0];
          vec.values.val[1] = vreinterpretq_f32_u32(mask_high);
          vec.values.val[1] = vbslq_f32(
              vreinterpretq_u32_f32(vec.values.val[1]),
              b.values.val[1],
              a.values.val[1]);
          return vec;
        }
    }
    return b;
  }
  static Vec256<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f32_x2(reinterpret_cast<const float*>(ptr));
    }
    else if (count == (size() >> 1)) {
      Vec256<float> res;
      res.values.val[0] = vld1q_f32(reinterpret_cast<const float*>(ptr));
      res.values.val[1] = vdupq_n_f32(0.f);
      return res;
    }
    else {
      __at_align32__ float tmp_values[size()];
      for (auto i = 0; i < size(); ++i) {
        tmp_values[i] = 0.0;
      }
      std::memcpy(
          tmp_values,
          reinterpret_cast<const float*>(ptr),
          count * sizeof(float));
      return vld1q_f32_x2(reinterpret_cast<const float*>(tmp_values));
    }
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f32_x2(reinterpret_cast<float*>(ptr), values);
    }
    else if (count == (size() >> 1)) {
      vst1q_f32(reinterpret_cast<float*>(ptr), values.val[0]);
    }
    else {
      float tmp_values[size()];
      vst1q_f32_x2(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float));
    }
  }
  inline const float32x4_t& get_low() const {
    return values.val[0];
  }
  inline float32x4_t& get_low() {
    return values.val[0];
  }
  inline const float32x4_t& get_high() const {
    return values.val[1];
  }
  inline float32x4_t& get_high() {
    return values.val[1];
  }
  // Very slow implementation of indexing.
  // Only required because vec256_qint refers to this.
  // Once we specialize that implementation for ARM
  // this should be removed. TODO (kimishpatel)
  const float operator[](int idx) const {
    __at_align32__ float tmp[size()];
    store(tmp);
    return tmp[idx];
  };
  const float operator[](int idx) {
    __at_align32__ float tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  // For boolean version where we want to if any 1/all zero
  // etc. can be done faster in a different way.
  int zero_mask() const {
    __at_align32__ float tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++ i) {
      if (tmp[i] == 0.f) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vec256<float> map(float (*f)(float)) const {
    __at_align32__ float tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> abs() const {
    return Vec256<float>(vabsq_f32(values.val[0]), vabsq_f32(values.val[1]));
  }
  Vec256<float> angle() const {
    return Vec256<float>(0.f);
  }
  Vec256<float> real() const {
    return *this;
  }
  Vec256<float> imag() const {
    return Vec256<float>(0.f);
  }
  Vec256<float> conj() const {
    return *this;
  }
  Vec256<float> acos() const {
    return map(std::acos);
  }
  Vec256<float> asin() const {
    return map(std::asin);
  }
  Vec256<float> atan() const {
    return map(std::atan);
  }
  Vec256<float> atan2(const Vec256<float> &exp) const {
    __at_align32__ float tmp[size()];
    __at_align32__ float tmp_exp[size()];
    store(tmp);
    exp.store(tmp_exp);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = std::atan2(tmp[i], tmp_exp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> erf() const {
    return map(std::erf);
  }
  Vec256<float> erfc() const {
    return map(std::erfc);
  }
  Vec256<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vec256<float> exp() const {
    return map(std::exp);
  }
  Vec256<float> expm1() const {
    return map(std::expm1);
  }
  Vec256<float> fmod(const Vec256<float>& q) const {
    __at_align32__ float tmp[size()];
    __at_align32__ float tmp_q[size()];
    store(tmp);
    q.store(tmp_q);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = std::fmod(tmp[i], tmp_q[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> hypot(const Vec256<float> &b) const {
    __at_align32__ float tmp[size()];
    __at_align32__ float tmp_b[size()];
    store(tmp);
    b.store(tmp_b);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = std::hypot(tmp[i], tmp_b[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> i0() const {
    return map(calc_i0);
  }
  Vec256<float> log() const {
    return map(std::log);
  }
  Vec256<float> log10() const {
    return map(std::log10);
  }
  Vec256<float> log1p() const {
    return map(std::log1p);
  }
  Vec256<float> log2() const {
    return map(std::log2);
  }
  Vec256<float> nextafter(const Vec256<float> &b) const {
    __at_align32__ float tmp[size()];
    __at_align32__ float tmp_b[size()];
    store(tmp);
    b.store(tmp_b);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> frac() const;
  Vec256<float> sin() const {
    return map(std::sin);
  }
  Vec256<float> sinh() const {
    return map(std::sinh);
  }
  Vec256<float> cos() const {
    return map(std::cos);
  }
  Vec256<float> cosh() const {
    return map(std::cosh);
  }
  Vec256<float> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vec256<float> floor() const {
    return map(at::native::floor_impl);
  }
  Vec256<float> neg() const {
    return Vec256<float>(
        vnegq_f32(values.val[0]),
        vnegq_f32(values.val[1]));
  }
  Vec256<float> round() const {
    // We do not use std::round because we would like to round midway numbers to the nearest even integer.
    return map(at::native::round_impl);
  }
  Vec256<float> tan() const {
    return map(std::tan);
  }
  Vec256<float> tanh() const {
    return map(std::tanh);
  }
  Vec256<float> trunc() const {
    float32x4_t r0 = vcvtq_f32_s32(vcvtq_s32_f32(values.val[0]));
    float32x4_t r1 = vcvtq_f32_s32(vcvtq_s32_f32(values.val[1]));
    return Vec256<float>(r0, r1);
  }
  Vec256<float> lgamma() const {
    return map(std::lgamma);
  }
  Vec256<float> sqrt() const {
    return Vec256<float>(
        vsqrtq_f32(values.val[0]),
        vsqrtq_f32(values.val[1]));
  }
  Vec256<float> reciprocal() const {
    return Vec256<float>(
        vrecpeq_f32(values.val[0]),
        vrecpeq_f32(values.val[1]));
  }
  Vec256<float> rsqrt() const {
    return Vec256<float>(
        vrsqrteq_f32(values.val[0]),
        vrsqrteq_f32(values.val[1]));
  }
  Vec256<float> pow(const Vec256<float> &exp) const {
    __at_align32__ float tmp[size()];
    __at_align32__ float tmp_exp[size()];
    store(tmp);
    exp.store(tmp_exp);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = std::pow(tmp[i], tmp_exp[i]);
    }
    return loadu(tmp);
  }
  Vec256<float> operator==(const Vec256<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vceqq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vceqq_f32(values.val[1], other.values.val[1]));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> operator!=(const Vec256<float>& other) const {
    float32x4_t r0 = vreinterpretq_f32_u32(
        vmvnq_u32(vceqq_f32(values.val[0], other.values.val[0])));
    float32x4_t r1 = vreinterpretq_f32_u32(
        vmvnq_u32(vceqq_f32(values.val[1], other.values.val[1])));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> operator<(const Vec256<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcltq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcltq_f32(values.val[1], other.values.val[1]));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> operator<=(const Vec256<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcleq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcleq_f32(values.val[1], other.values.val[1]));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> operator>(const Vec256<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcgtq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcgtq_f32(values.val[1], other.values.val[1]));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> operator>=(const Vec256<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcgeq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcgeq_f32(values.val[1], other.values.val[1]));
    return Vec256<float>(r0, r1);
  }

  Vec256<float> eq(const Vec256<float>& other) const;
  Vec256<float> ne(const Vec256<float>& other) const;
  Vec256<float> gt(const Vec256<float>& other) const;
  Vec256<float> ge(const Vec256<float>& other) const;
  Vec256<float> lt(const Vec256<float>& other) const;
  Vec256<float> le(const Vec256<float>& other) const;
};

template <>
Vec256<float> inline operator+(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vaddq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vaddq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline operator-(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vsubq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vsubq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline operator*(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vmulq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmulq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline operator/(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vdivq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vdivq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

// frac. Implement this here so we can use subtraction
Vec256<float> Vec256<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline maximum(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vmaxq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmaxq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vec256<float> inline minimum(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vminq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vminq_f32(a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline clamp(const Vec256<float>& a, const Vec256<float>& min, const Vec256<float>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vec256<float> inline clamp_max(const Vec256<float>& a, const Vec256<float>& max) {
  return minimum(max, a);
}

template <>
Vec256<float> inline clamp_min(const Vec256<float>& a, const Vec256<float>& min) {
  return maximum(min, a);
}

template <>
Vec256<float> inline operator&(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline operator|(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vec256<float>(r0, r1);
}

template <>
Vec256<float> inline operator^(const Vec256<float>& a, const Vec256<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vec256<float>(r0, r1);
}

Vec256<float> Vec256<float>::eq(const Vec256<float>& other) const {
  return (*this == other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::ne(const Vec256<float>& other) const {
  return (*this != other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::gt(const Vec256<float>& other) const {
  return (*this > other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::ge(const Vec256<float>& other) const {
  return (*this >= other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::lt(const Vec256<float>& other) const {
  return (*this < other) & Vec256<float>(1.0f);
}

Vec256<float> Vec256<float>::le(const Vec256<float>& other) const {
  return (*this <= other) & Vec256<float>(1.0f);
}

template <>
inline void convert(const float* src, int32_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vec256<float>::size()); i += Vec256<float>::size()) {
    vst1q_s32(dst + i, vcvtq_s32_f32(vld1q_f32(src + i)));
    vst1q_s32(dst + i + 4, vcvtq_s32_f32(vld1q_f32(src + i + 4)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<int32_t>(src[i]);
  }
}

template <>
inline void convert(const int32_t* src, float* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vec256<float>::size()); i += Vec256<float>::size()) {
    vst1q_f32(dst + i, vcvtq_f32_s32(vld1q_s32(src + i)));
    vst1q_f32(dst + i + 4, vcvtq_f32_s32(vld1q_s32(src + i + 4)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
Vec256<float> inline fmadd(const Vec256<float>& a, const Vec256<float>& b, const Vec256<float>& c) {
  float32x4_t r0 = vfmaq_f32(c.get_low(), a.get_low(), b.get_low());
  float32x4_t r1 = vfmaq_f32(c.get_high(), a.get_high(), b.get_high());
  return Vec256<float>(r0, r1);
}

#endif

}}}
