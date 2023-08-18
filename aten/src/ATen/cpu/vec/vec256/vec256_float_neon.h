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

namespace at {
namespace vec {
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

template <> class Vectorized<float> {
private:
  float32x4x2_t values;
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return 8;
  }
  Vectorized() {}
  Vectorized(float32x4x2_t v) : values(v) {}
  Vectorized(float val) : values{vdupq_n_f32(val), vdupq_n_f32(val) } {}
  Vectorized(float val0, float val1, float val2, float val3,
         float val4, float val5, float val6, float val7) :
         values{val0, val1, val2, val3, val4, val5, val6, val7} {}
  Vectorized(float32x4_t val0, float32x4_t val1) : values{val0, val1} {}
  operator float32x4x2_t() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b) {
    Vectorized<float> vec;
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
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask) {
    // TODO
    // NB: This requires that each value, i.e., each uint value,
    // of the mask either all be zeros or all be 1s.
    // We perhaps need some kind of an assert?
    // But that will affect performance.
    Vectorized<float> vec(mask.values);
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
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    const Vectorized<float> base_vec(base);
    const Vectorized<float> step_vec(step);
    const Vectorized<float> step_sizes(0, 1, 2, 3, 4, 5, 6, 7);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        {
          Vectorized<float> vec;
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
          Vectorized<float> vec;
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
          Vectorized<float> vec;
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
        return Vectorized<float>(b.values.val[0], a.values.val[1]);
      case 5:
        {
          Vectorized<float> vec;
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
          Vectorized<float> vec;
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
          Vectorized<float> vec;
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
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size()) {
      return vld1q_f32_x2(reinterpret_cast<const float*>(ptr));
    }
    else if (count == (size() >> 1)) {
      Vectorized<float> res;
      res.values.val[0] = vld1q_f32(reinterpret_cast<const float*>(ptr));
      res.values.val[1] = vdupq_n_f32(0.f);
      return res;
    }
    else {
      __at_align__ float tmp_values[size()];
      for (const auto i : c10::irange(size())) {
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
    for (int i = 0; i < size(); ++ i) {
      if (tmp[i] == 0.f) {
        mask |= (1 << i);
      }
    }
    return mask;
  }
  Vectorized<float> isnan() const {
    __at_align__ float tmp[size()];
    __at_align__ float res[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      if (_isnan(tmp[i])) {
        std::memset(static_cast<void*>(&res[i]), 0xFF, sizeof(float));
      } else {
        std::memset(static_cast<void*>(&res[i]), 0, sizeof(float));
      }
    }
    return loadu(res);
  };
  Vectorized<float> map(float (*const f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    return Vectorized<float>(vabsq_f32(values.val[0]), vabsq_f32(values.val[1]));
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
  Vectorized<float> acos() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_acosf4_u10(values.val[0]), Sleef_acosf4_u10(values.val[1])),
      map(std::acos)
    );
  }
  Vectorized<float> acosh() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_acoshf4_u10(values.val[0]), Sleef_acoshf4_u10(values.val[1])),
      map(std::acosh)
    );
  }
  Vectorized<float> asin() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_asinf4_u10(values.val[0]), Sleef_asinf4_u10(values.val[1])),
      map(std::asin)
    );
  }
  Vectorized<float> atan() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_atanf4_u10(values.val[0]), Sleef_atanf4_u10(values.val[1])),
      map(std::atan)
    );
  }
  Vectorized<float> atan2(const Vectorized<float> &exp) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_atan2f4_u10(values.val[0], exp.values.val[0]),
                                 Sleef_atan2f4_u10(values.val[1], exp.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_exp[size()];
        store(tmp);
        exp.store(tmp_exp);
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::atan2(tmp[i], tmp_exp[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_copysignf4(values.val[0], sign.values.val[0]),
                                 Sleef_copysignf4(values.val[1], sign.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_sign[size()];
        store(tmp);
        sign.store(tmp_sign);
        for (size_type i = 0; i < size(); i++) {
          tmp[i] = std::copysign(tmp[i], tmp_sign[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> erf() const;
  Vectorized<float> erfc() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_erfcf4_u15(values.val[0]), Sleef_erfcf4_u15(values.val[1])),
      map(std::erfc)
    );
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_expf4_u10(values.val[0]), Sleef_expf4_u10(values.val[1])),
      map(std::exp)
    );
  }
  Vectorized<float> exp2() const {
    return USE_SLEEF(
        Vectorized<float>(Sleef_exp2f4_u10(values.val[0]), Sleef_exp2f4_u10(values.val[1])),
        map(std::exp2)
      );
  }
  Vectorized<float> expm1() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_expm1f4_u10(values.val[0]), Sleef_expm1f4_u10(values.val[1])),
      map(std::expm1)
    );
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_fmodf4(values.val[0], q.values.val[0]),
                                 Sleef_fmodf4(values.val[1], q.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_q[size()];
        store(tmp);
        q.store(tmp_q);
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::fmod(tmp[i], tmp_q[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_hypotf4_u05(values.val[0], b.values.val[0]),
                                 Sleef_hypotf4_u05(values.val[1], b.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::hypot(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> i0() const {
    return map(calc_i0);
  }
  Vectorized<float> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (const auto i : c10::irange(size())) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> log() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_logf4_u10(values.val[0]), Sleef_logf4_u10(values.val[1])),
      map(std::log)
    );
  }
  Vectorized<float> log10() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_log10f4_u10(values.val[0]), Sleef_log10f4_u10(values.val[1])),
      map(std::log10)
    );
  }
  Vectorized<float> log1p() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_log1pf4_u10(values.val[0]), Sleef_log1pf4_u10(values.val[1])),
      map(std::log1p)
    );
  }
  Vectorized<float> log2() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_log2f4_u10(values.val[0]), Sleef_log2f4_u10(values.val[1])),
      map(std::log2)
    );
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_nextafterf4(values.val[0], b.values.val[0]),
                                 Sleef_nextafterf4(values.val[1], b.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_sinf4_u10(values.val[0]), Sleef_sinf4_u10(values.val[1])),
      map(std::sin)
    );
  }
  Vectorized<float> sinh() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_sinhf4_u10(values.val[0]), Sleef_sinhf4_u10(values.val[1])),
      map(std::sinh)
    );
  }
  Vectorized<float> cos() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_cosf4_u10(values.val[0]), Sleef_cosf4_u10(values.val[1])),
      map(std::cos)
    );
  }
  Vectorized<float> cosh() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_coshf4_u10(values.val[0]), Sleef_coshf4_u10(values.val[1])),
      map(std::cosh)
    );
  }
  Vectorized<float> ceil() const {
    return map(at::native::ceil_impl);
  }
  Vectorized<float> floor() const {
    return map(at::native::floor_impl);
  }
  Vectorized<float> neg() const {
    return Vectorized<float>(
        vnegq_f32(values.val[0]),
        vnegq_f32(values.val[1]));
  }
  Vectorized<float> round() const {
    // We do not use std::round because we would like to round midway numbers to the nearest even integer.
    return map(at::native::round_impl);
  }
  Vectorized<float> tan() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_tanf4_u10(values.val[0]), Sleef_tanf4_u10(values.val[1])),
      map(std::tan)
    );
  }
  Vectorized<float> tanh() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_tanhf4_u10(values.val[0]), Sleef_tanhf4_u10(values.val[1])),
      map(std::tanh)
    );
  }
  Vectorized<float> trunc() const {
    float32x4_t r0 = vrndq_f32(values.val[0]);
    float32x4_t r1 = vrndq_f32(values.val[1]);
    return Vectorized<float>(r0, r1);
  }
  Vectorized<float> lgamma() const {
    return USE_SLEEF(
      Vectorized<float>(Sleef_lgammaf4_u10(values.val[0]), Sleef_lgammaf4_u10(values.val[1])),
      map(std::lgamma)
    );
  }
  Vectorized<float> sqrt() const {
    return Vectorized<float>(
        vsqrtq_f32(values.val[0]),
        vsqrtq_f32(values.val[1]));
  }
  Vectorized<float> reciprocal() const {
    auto r0 = vdivq_f32(vdupq_n_f32(1.0f), values.val[0]);
    auto r1 = vdivq_f32(vdupq_n_f32(1.0f), values.val[1]);
    return Vectorized<float>(r0, r1);
  }
  Vectorized<float> rsqrt() const {
    return this->sqrt().reciprocal();
  }
  Vectorized<float> pow(const Vectorized<float> &exp) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_powf4_u10(values.val[0], exp.values.val[0]),
                                 Sleef_powf4_u10(values.val[1], exp.values.val[1]));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_exp[size()];
        store(tmp);
        exp.store(tmp_exp);
        for (const auto i : c10::irange(size())) {
          tmp[i] = std::pow(tmp[i], tmp_exp[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vceqq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vceqq_f32(values.val[1], other.values.val[1]));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    float32x4_t r0 = vreinterpretq_f32_u32(
        vmvnq_u32(vceqq_f32(values.val[0], other.values.val[0])));
    float32x4_t r1 = vreinterpretq_f32_u32(
        vmvnq_u32(vceqq_f32(values.val[1], other.values.val[1])));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcltq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcltq_f32(values.val[1], other.values.val[1]));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcleq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcleq_f32(values.val[1], other.values.val[1]));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcgtq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcgtq_f32(values.val[1], other.values.val[1]));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    float32x4_t r0 =
      vreinterpretq_f32_u32(vcgeq_f32(values.val[0], other.values.val[0]));
    float32x4_t r1 =
      vreinterpretq_f32_u32(vcgeq_f32(values.val[1], other.values.val[1]));
    return Vectorized<float>(r0, r1);
  }

  Vectorized<float> eq(const Vectorized<float>& other) const;
  Vectorized<float> ne(const Vectorized<float>& other) const;
  Vectorized<float> gt(const Vectorized<float>& other) const;
  Vectorized<float> ge(const Vectorized<float>& other) const;
  Vectorized<float> lt(const Vectorized<float>& other) const;
  Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
Vectorized<float> inline operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vaddq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vaddq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vsubq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vsubq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vmulq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmulq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vdivq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vdivq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vmaxq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vmaxq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vminq_f32(a.get_low(), b.get_low());
  float32x4_t r1 = vminq_f32(a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return minimum(max, maximum(min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return minimum(max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return maximum(min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(vandq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(vorrq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  float32x4_t r0 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_low()),
      vreinterpretq_u32_f32(b.get_low())));
  float32x4_t r1 = vreinterpretq_f32_u32(veorq_u32(
      vreinterpretq_u32_f32(a.get_high()),
      vreinterpretq_u32_f32(b.get_high())));
  return Vectorized<float>(r0, r1);
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, int32_t* dst, int64_t n) {
  int64_t i;
#pragma unroll
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
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
  for (i = 0; i <= (n - Vectorized<float>::size()); i += Vectorized<float>::size()) {
    vst1q_f32(dst + i, vcvtq_f32_s32(vld1q_s32(src + i)));
    vst1q_f32(dst + i + 4, vcvtq_f32_s32(vld1q_s32(src + i + 4)));
  }
#pragma unroll
  for (; i < n; i++) {
    dst[i] = static_cast<float>(src[i]);
  }
}

template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  float32x4_t r0 = vfmaq_f32(c.get_low(), a.get_low(), b.get_low());
  float32x4_t r1 = vfmaq_f32(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

template <>
Vectorized<float> inline fmsub(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  float32x4_t r0 = vfmsq_f32(c.get_low(), a.get_low(), b.get_low());
  float32x4_t r1 = vfmsq_f32(c.get_high(), a.get_high(), b.get_high());
  return Vectorized<float>(r0, r1);
}

inline Vectorized<float> Vectorized<float>::erf() const{
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
    auto tmp4 = neg_pow_2.map(std::exp); // This can be swapped for a faster implementation of exp.
    auto tmp5 = tmp4 ^ neg_zero_vec;
    // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
    auto tmp6 = t * tmp5;
    auto tmp7 = fmadd(tmp6, r, one_vec);
    return tmp7 ^ sign_mask;
}
#endif /* defined(aarch64) */

}}}
