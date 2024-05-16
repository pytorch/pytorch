#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec256/vec256_float_neon.h>
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
#if !defined(C10_MOBILE) && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

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

// On ARM, Half type supports float16_t->Half constructor and Half->float16_t
// conversion
template <>
class Vectorized<c10::Half> {
 private:
  float16x8x2_t values;

 public:
  // value_type should be c10::Half to fit interface with vec_base.h
  using value_type = c10::Half;
  using size_type = int;
  static constexpr size_type size() {
    static_assert(sizeof(float16x8x2_t) == 16 * sizeof(value_type));
    return 16;
  }

 private:
  // We use these private map functions to implement various methods
  Vectorized<c10::Half> map2(
      const Vectorized<c10::Half>& second,
      c10::Half (*const f)(c10::Half, c10::Half)) const {
    __at_align__ c10::Half tmp_first[size()];
    __at_align__ c10::Half tmp_second[size()];
    store(tmp_first); // store this to tmp_first
    second.store(tmp_second);
    for (const auto i : c10::irange(size())) {
      tmp_first[i] = f(tmp_first[i], tmp_second[i]);
    }
    return loadu(tmp_first);
  }

  Vectorized<c10::Half> map_with_vec_float_method(
      Vectorized<float> (Vectorized<float>::*m)() const) const {
    // Convert low float16x8_t to 2 float32x4_t variables, apply m, and convert
    // back
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values.val[0]));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values.val[0]));
    Vectorized<float> mv0 = (Vectorized<float>(v00, v01).*m)();
    float16x4_t r00 = vcvt_f16_f32(mv0.get_low());
    float16x4_t r01 = vcvt_f16_f32(mv0.get_high());

    // Convert high float16x8_t to 2 float32x4_t variables, apply m, and convert
    // back
    float32x4_t v10 = vcvt_f32_f16(vget_low_f16(values.val[1]));
    float32x4_t v11 = vcvt_f32_f16(vget_high_f16(values.val[1]));
    Vectorized<float> mv1 = (Vectorized<float>(v10, v11).*m)();
    float16x4_t r10 = vcvt_f16_f32(mv1.get_low());
    float16x4_t r11 = vcvt_f16_f32(mv1.get_high());

    // Pack result into Vectorized<c10::Half>
    return Vectorized<c10::Half>(
        vcombine_f16(r00, r01), vcombine_f16(r10, r11));
  }

  Vectorized<c10::Half> map2_with_vec_float_method(
      const Vectorized<c10::Half>& second,
      Vectorized<float> (Vectorized<float>::*m)(const Vectorized<float>&)
          const) const {
    // Convert low float16x8_t to 2 float32x4_t variables, apply m, and convert
    // back
    float32x4_t v00 = vcvt_f32_f16(vget_low_f16(values.val[0]));
    float32x4_t v01 = vcvt_f32_f16(vget_high_f16(values.val[0]));
    float32x4_t second_v00 = vcvt_f32_f16(vget_low_f16(second.get_low()));
    float32x4_t second_v01 = vcvt_f32_f16(vget_high_f16(second.get_low()));
    Vectorized<float> mv0 = (Vectorized<float>(v00, v01).*m)(
        Vectorized<float>(second_v00, second_v01));
    float16x4_t r00 = vcvt_f16_f32(mv0.get_low());
    float16x4_t r01 = vcvt_f16_f32(mv0.get_high());

    // Convert high float16x8_t to 2 float32x4_t variables, apply m, and convert
    // back
    float32x4_t v10 = vcvt_f32_f16(vget_low_f16(values.val[1]));
    float32x4_t v11 = vcvt_f32_f16(vget_high_f16(values.val[1]));
    float32x4_t second_v10 = vcvt_f32_f16(vget_low_f16(second.get_high()));
    float32x4_t second_v11 = vcvt_f32_f16(vget_high_f16(second.get_high()));
    Vectorized<float> mv1 = (Vectorized<float>(v10, v11).*m)(
        Vectorized<float>(second_v10, second_v11));
    float16x4_t r10 = vcvt_f16_f32(mv1.get_low());
    float16x4_t r11 = vcvt_f16_f32(mv1.get_high());

    // Pack result into Vectorized<c10::Half>
    return Vectorized<c10::Half>(
        vcombine_f16(r00, r01), vcombine_f16(r10, r11));
  }

 public:
   // constructor
  Vectorized() {}
  Vectorized(float16x8x2_t v) : values(v) {}

  // A ctor that accepts c10::Half is needed to fit interface with vec_base.h
  // A second constructor that takes float16_t is also included
  Vectorized(c10::Half val)
      : values{vdupq_n_f16((float16_t)val), vdupq_n_f16((float16_t)val)} {
  }
  Vectorized(float16_t val) : values{vdupq_n_f16(val), vdupq_n_f16(val)} {}
  Vectorized(
      float16_t val0,
      float16_t val1,
      float16_t val2,
      float16_t val3,
      float16_t val4,
      float16_t val5,
      float16_t val6,
      float16_t val7,
      float16_t val8,
      float16_t val9,
      float16_t val10,
      float16_t val11,
      float16_t val12,
      float16_t val13,
      float16_t val14,
      float16_t val15)
      : values{
            val0,
            val1,
            val2,
            val3,
            val4,
            val5,
            val6,
            val7,
            val8,
            val9,
            val10,
            val11,
            val12,
            val13,
            val14,
            val15} {}
  Vectorized(float16x8_t val0, float16x8_t val1) : values{val0, val1} {}
  operator float16x8x2_t() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<c10::Half> blend(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b) {
    Vectorized<c10::Half> vec;
    // 0.
    vec.values.val[0] = BlendHalfRegs<0, (mask & 0x01) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<1, (mask & 0x02) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<2, (mask & 0x04) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<3, (mask & 0x08) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);

    vec.values.val[0] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);
    vec.values.val[0] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
        a.values.val[0], b.values.val[0], vec.values.val[0]);

    // 1.
    vec.values.val[1] = BlendHalfRegs<0, (mask & 0x10) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<1, (mask & 0x20) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<2, (mask & 0x40) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<3, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    vec.values.val[1] = BlendHalfRegs<4, (mask & 0x10) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<5, (mask & 0x20) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<6, (mask & 0x40) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);
    vec.values.val[1] = BlendHalfRegs<7, (mask & 0x80) != 0>::impl(
        a.values.val[1], b.values.val[1], vec.values.val[1]);

    return vec;
  }
  static Vectorized<c10::Half> blendv(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      const Vectorized<c10::Half>& mask) {
    // Note: using blendv is very awkward because 0xFFFF is one of many NaN's in
    // FP16 It's unfortunate that the mask has type Half (required from
    // vec_base)

    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    Vectorized<c10::Half> vec(mask.values);
    vec.values.val[0] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[0]),
        b.values.val[0],
        a.values.val[0]);
    vec.values.val[1] = vbslq_f16(
        vreinterpretq_u16_f16(vec.values.val[1]),
        b.values.val[1],
        a.values.val[1]);
    return vec;
  }
  template <typename step_t>
  static Vectorized<c10::Half> arange(
      c10::Half base = 0.0,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<c10::Half> base_vec(base);
    const Vectorized<c10::Half> step_vec(step);
    const Vectorized<c10::Half> step_sizes(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<c10::Half> set(
      const Vectorized<c10::Half>& a,
      const Vectorized<c10::Half>& b,
      int64_t count = size()) {
    uint16_t pre_mask[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < count; i++) {
      pre_mask[i] = 0xFFFF;
    }
    uint16x8x2_t mask = vld1q_u16_x2(pre_mask);

    // Using blendv is awkward because 0xFFFF is one of many NaN's in FP16
    // so we directly use vbslq_f16 instead
    Vectorized<c10::Half> vec(
        vbslq_f16(
            // Low bits
            mask.val[0],
            b.values.val[0],
            a.values.val[0]),
        // High bits
        vbslq_f16(mask.val[1], b.values.val[1], a.values.val[1]));

    return vec;
  }
  static Vectorized<c10::Half> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f16_x2(reinterpret_cast<const float16_t*>(ptr));
    } else if (count == (size() >> 1)) {
      Vectorized<c10::Half> res;
      res.values.val[0] = vld1q_f16(reinterpret_cast<const float16_t*>(ptr));
      std::memset(&res.values.val[1], 0, sizeof(res.values.val[1]));
      return res;
    }
    __at_align__ float16_t tmp_values[size()];
    for (const auto i : c10::irange(size())) {
      tmp_values[i] = 0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float16_t*>(ptr),
        count * sizeof(float16_t));
    return vld1q_f16_x2(reinterpret_cast<const float16_t*>(tmp_values));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f16_x2(reinterpret_cast<float16_t*>(ptr), values);
      return;
    } else if (count == (size() >> 1)) {
      vst1q_f16(reinterpret_cast<float16_t*>(ptr), values.val[0]);
    } else {
      float16_t tmp_values[size()];
      vst1q_f16_x2(reinterpret_cast<float16_t*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(float16_t));
    }
  }
  inline const float16x8_t& get_low() const {
    return values.val[0];
  }
  inline float16x8_t& get_low() {
    return values.val[0];
  }
  inline const float16x8_t& get_high() const {
    return values.val[1];
  }
  inline float16x8_t& get_high() {
    return values.val[1];
  }
  // Very slow implementation of indexing.
  // Only required because vec256_qint refers to this.
  // Once we specialize that implementation for ARM
  // this should be removed. TODO (kimishpatel)
  c10::Half operator[](int idx) const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  c10::Half operator[](int idx) {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    return tmp[idx];
  }
  // For boolean version where we want to if any 1/all zero
  // etc. can be done faster in a different way.
  int zero_mask() const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    int mask = 0;
    for (int i = 0; i < size(); ++i) {
      if (tmp[i] == 0) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<c10::Half> isnan() const {
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
  };
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
  Vectorized<c10::Half> map(c10::Half (*const f)(c10::Half)) const {
    __at_align__ c10::Half tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<c10::Half> abs() const {
    return Vectorized<c10::Half>(
        vabsq_f16(values.val[0]), vabsq_f16(values.val[1]));
  }
  Vectorized<c10::Half> angle() const {
    auto zero = Vectorized<c10::Half>(0);
    auto pi = Vectorized<c10::Half>(c10::pi<c10::Half>);
    auto tmp = blendv(zero, pi, *this < zero);
    return blendv(tmp, *this, isnan());
  }
  Vectorized<c10::Half> real() const {
    return *this;
  }
  Vectorized<c10::Half> imag() const {
    return Vectorized<c10::Half>(0);
  }
  Vectorized<c10::Half> conj() const {
    return *this;
  }

  // Sleef does not support FP16, so many math functions are applied by
  // converting to FP32, applying the math function, and then converting back to
  // FP16.
  Vectorized<c10::Half> acos() const {
    return map_with_vec_float_method(&Vectorized<float>::acos);
  }
  Vectorized<c10::Half> acosh() const {
    return map_with_vec_float_method(&Vectorized<float>::acosh);
  }
  Vectorized<c10::Half> asin() const {
    return map_with_vec_float_method(&Vectorized<float>::asin);
  }
  Vectorized<c10::Half> atan() const {
    return map_with_vec_float_method(&Vectorized<float>::atan);
  }
  Vectorized<c10::Half> atanh() const {
    return map_with_vec_float_method(&Vectorized<float>::atanh);
  }
  Vectorized<c10::Half> atan2(const Vectorized<c10::Half>& exp) const {
    return map2_with_vec_float_method(exp, &Vectorized<float>::atan2);
  }
  Vectorized<c10::Half> copysign(const Vectorized<c10::Half>& sign) const {
    return map2_with_vec_float_method(sign, &Vectorized<float>::copysign);
  }
  Vectorized<c10::Half> erf() const {
    return map_with_vec_float_method(&Vectorized<float>::erf);
  }
  Vectorized<c10::Half> erfc() const {
    return map_with_vec_float_method(&Vectorized<float>::erfc);
  }
  Vectorized<c10::Half> erfinv() const {
    return map_with_vec_float_method(&Vectorized<float>::erfinv);
  }
  Vectorized<c10::Half> exp() const {
    return map_with_vec_float_method(&Vectorized<float>::exp);
  }
  Vectorized<c10::Half> exp2() const {
    return map_with_vec_float_method(&Vectorized<float>::exp2);
  }
  Vectorized<c10::Half> expm1() const {
    return map_with_vec_float_method(&Vectorized<float>::expm1);
  }
  Vectorized<c10::Half> exp_u20() const {
    return map_with_vec_float_method(&Vectorized<float>::exp_u20);
  }
  Vectorized<c10::Half> fmod(const Vectorized<c10::Half>& q) const {
    // This function is questionable with a conversion, so we use map2
    return map2(q, std::fmod);
  }
  Vectorized<c10::Half> hypot(const Vectorized<c10::Half>& b) const {
    return map2_with_vec_float_method(b, &Vectorized<float>::hypot);
  }
  Vectorized<c10::Half> i0() const {
    return map_with_vec_float_method(&Vectorized<float>::i0);
  }
  Vectorized<c10::Half> i0e() const {
    return map_with_vec_float_method(&Vectorized<float>::i0e);
  }
  Vectorized<c10::Half> digamma() const {
    return map_with_vec_float_method(&Vectorized<float>::digamma);
  }
  Vectorized<c10::Half> igamma(const Vectorized<c10::Half>& x) const {
    return map2_with_vec_float_method(x, &Vectorized<float>::igamma);
  }
  Vectorized<c10::Half> igammac(const Vectorized<c10::Half>& x) const {
    return map2_with_vec_float_method(x, &Vectorized<float>::igammac);
  }
  Vectorized<c10::Half> log() const {
    return map_with_vec_float_method(&Vectorized<float>::log);
  }
  Vectorized<c10::Half> log10() const {
    return map_with_vec_float_method(&Vectorized<float>::log10);
  }
  Vectorized<c10::Half> log1p() const {
    return map_with_vec_float_method(&Vectorized<float>::log1p);
  }
  Vectorized<c10::Half> log2() const {
    return map_with_vec_float_method(&Vectorized<float>::log2);
  }
  Vectorized<c10::Half> nextafter(const Vectorized<c10::Half>& b) const {
    // This function does not make sense with conversion, so we use map2
    return map2(b, std::nextafter);
  }
  Vectorized<c10::Half> frac() const;
  Vectorized<c10::Half> sin() const {
    return map_with_vec_float_method(&Vectorized<float>::sin);
  }
  Vectorized<c10::Half> sinh() const {
    return map_with_vec_float_method(&Vectorized<float>::sinh);
  }
  Vectorized<c10::Half> cos() const {
    return map_with_vec_float_method(&Vectorized<float>::cos);
  }
  Vectorized<c10::Half> cosh() const {
    return map_with_vec_float_method(&Vectorized<float>::cosh);
  }
  Vectorized<c10::Half> ceil() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::ceil_impl);
  }
  Vectorized<c10::Half> floor() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::floor_impl);
  }
  Vectorized<c10::Half> neg() const {
    return Vectorized<c10::Half>(
        vnegq_f16(values.val[0]), vnegq_f16(values.val[1]));
  }
  inline Vectorized<c10::Half> round() const {
    // This function is questionable with a conversion, so we use map
    return map(at::native::round_impl);
  }
  inline Vectorized<c10::Half> tan() const {
    return map_with_vec_float_method(&Vectorized<float>::tan);
  }
  inline Vectorized<c10::Half> tanh() const {
    return map_with_vec_float_method(&Vectorized<float>::tanh);
  }
  Vectorized<c10::Half> trunc() const {
    float16x8_t r0 = vrndq_f16(values.val[0]);
    float16x8_t r1 = vrndq_f16(values.val[1]);
    return Vectorized<c10::Half>(r0, r1);
  }
  Vectorized<c10::Half> lgamma() const {
    return map_with_vec_float_method(&Vectorized<float>::lgamma);
  }
  Vectorized<c10::Half> sqrt() const {
    return Vectorized<c10::Half>(
        vsqrtq_f16(values.val[0]), vsqrtq_f16(values.val[1]));
  }
  Vectorized<c10::Half> reciprocal() const {
    auto ones = vdupq_n_f16(1.0f);
    auto r0 = vdivq_f16(ones, values.val[0]);
    auto r1 = vdivq_f16(ones, values.val[1]);
    return Vectorized<c10::Half>(r0, r1);
  }
  Vectorized<c10::Half> rsqrt() const {
    return this->sqrt().reciprocal();
  }
  Vectorized<c10::Half> pow(const Vectorized<c10::Half>& exp) const {
    return map2_with_vec_float_method(exp, &Vectorized<float>::pow);
  }
  Vectorized<c10::Half> operator==(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vceqq_f16(values.val[1], other.values.val[1]));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator!=(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[0], other.values.val[0])));
    float16x8_t r1 = vreinterpretq_f16_u16(
        vmvnq_u16(vceqq_f16(values.val[1], other.values.val[1])));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator<(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcltq_f16(values.val[1], other.values.val[1]));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator<=(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcleq_f16(values.val[1], other.values.val[1]));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator>(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgtq_f16(values.val[1], other.values.val[1]));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> operator>=(const Vectorized<c10::Half>& other) const {
    float16x8_t r0 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[0], other.values.val[0]));
    float16x8_t r1 =
        vreinterpretq_f16_u16(vcgeq_f16(values.val[1], other.values.val[1]));
    return Vectorized<c10::Half>(r0, r1);
  }

  Vectorized<c10::Half> eq(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> ne(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> gt(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> ge(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> lt(const Vectorized<c10::Half>& other) const;
  Vectorized<c10::Half> le(const Vectorized<c10::Half>& other) const;
}; // Vectorized<Half>

template <>
Vectorized<c10::Half> inline operator+(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vaddq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vaddq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline operator-(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vsubq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vsubq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline operator*(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vmulq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmulq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline operator/(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vdivq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vdivq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
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
  float16x8_t r0 = vmaxq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vmaxq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<c10::Half> inline minimum(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vminq_f16(a.get_low(), b.get_low());
  float16x8_t r1 = vminq_f16(a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
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
  float16x8_t r0 = vreinterpretq_f16_u16(vandq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(vandq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline operator|(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(vorrq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline operator^(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b) {
  float16x8_t r0 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_low()), vreinterpretq_u16_f16(b.get_low())));
  float16x8_t r1 = vreinterpretq_f16_u16(veorq_u16(
      vreinterpretq_u16_f16(a.get_high()),
      vreinterpretq_u16_f16(b.get_high())));
  return Vectorized<c10::Half>(r0, r1);
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

template <>
inline void convert(const float16_t* src, int16_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    vst1q_s16(dst + i, vcvtq_s16_f16(vld1q_f16(src + i)));
    vst1q_s16(dst + i + 8, vcvtq_s16_f16(vld1q_f16(src + i + 8)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<int16_t>(src[i]);
  }
}

template <>
inline void convert(const int16_t* src, float16_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<c10::Half>::size());
       i += Vectorized<c10::Half>::size()) {
    vst1q_f16(dst + i, vcvtq_f16_s16(vld1q_s16(src + i)));
    vst1q_f16(dst + i + 8, vcvtq_f16_s16(vld1q_s16(src + i + 8)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<float16_t>(src[i]);
  }
}

template <>
Vectorized<c10::Half> inline fmadd(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  float16x8_t r0 = vfmaq_f16(c.get_low(), a.get_low(), b.get_low());
  float16x8_t r1 = vfmaq_f16(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

template <>
Vectorized<c10::Half> inline fmsub(
    const Vectorized<c10::Half>& a,
    const Vectorized<c10::Half>& b,
    const Vectorized<c10::Half>& c) {
  float16x8_t r0 = vfmsq_f16(c.get_low(), a.get_low(), b.get_low());
  float16x8_t r1 = vfmsq_f16(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<c10::Half>(r0, r1);
}

#endif /* defined(aarch64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(C10_MOBILE) */

} // namespace CPU_CAPABILITY
} // namespace at::vec
