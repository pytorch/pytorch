#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>
#include <cmath>

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

#if defined(CPU_CAPABILITY_SVE128)

template <>
struct is_vec_specialized_for<double> : std::bool_constant<true> {};

template <>
class Vectorized<double> {
 private:
  float64x2_t values;

 public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return 2;
  }
  Vectorized() {
    values = vdupq_n_f64(0.0);
  }
  Vectorized(float64x2_t v) : values(v) {}
  Vectorized(svfloat64_t v) : values(svget_neonq(v)) {}
  Vectorized(double val) {
    values = svget_neonq(svdup_n_f64(val));
  }
  template <
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ double buffer[size()] = {vals...};
    values = vld1q_f64(buffer);
  }
  operator float64x2_t() const {
    return values;
  }
  operator svfloat64_t() const {
    return svset_neonq(svundef_f64(), values);
  }
  template <int64_t mask>
  static Vectorized<double> blend(
      const Vectorized<double>& a,
      const Vectorized<double>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint64x2_t maskArray = {
        (mask & 1ULL) ? 0xFFFFFFFFFFFFFFFF : 0,
        (mask & 2ULL) ? 0xFFFFFFFFFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_f64(maskArray, b.values, a.values);
  }
  static Vectorized<double> blendv(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      const Vectorized<double>& mask_) {
    return vbslq_f64(vreinterpretq_u64_f64(mask_.values), b.values, a.values);
  }
  template <typename step_t>
  static Vectorized<double> arange(
      double base = 0.,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<double> base_vec(base);
    const Vectorized<double> step_vec(step);
    const Vectorized<double> step_sizes = svreinterpret_f64_u64(
        svindex_u64(0, 4607182418800017408)); // {0.0, 1.0}
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static inline Vectorized<double> set(
      const Vectorized<double>& a,
      const Vectorized<double>& b,
      int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f64(svwhilelt_b64(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<double> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return vld1q_f64(reinterpret_cast<const double*>(ptr));
    svbool_t pg = svwhilelt_b64(0ull, count);
    return svld1_f64(pg, reinterpret_cast<const double*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f64(reinterpret_cast<double*>(ptr), values);
    } else {
      svbool_t pg = svwhilelt_b64(0ull, count);
      svst1_f64(
          pg,
          reinterpret_cast<double*>(ptr),
          svset_neonq(svundef_f64(), values));
    }
  }
  const double& operator[](int idx) const = delete;
  double& operator[](int idx) = delete;
  int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit
    // and others are translated to 0-bit
    uint64x2_t cmpReg = vceqzq_f64(values);
    uint32x2_t narrowedCmp = vmovn_u64(cmpReg);
    uint64x2_t extReg = svget_neonq(svbext_u64(
        svset_neonq(
            svundef_u64(),
            vreinterpretq_u64_u32(vcombine_u32(narrowedCmp, vdup_n_u32(0)))),
        svreinterpret_u64_u32(svdup_u32(1))));
    return extReg[0];
  }
  Vectorized<double> isnan() const {
    // NaN check
    return vreinterpretq_f64_u32(
        vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(values, values))));
  }
  bool has_inf_nan() const {
    return svptest_any(
        ptrue,
        svcmpuo_f64(
            ptrue,
            svset_neonq(svundef_f64(), vsubq_f64(values, values)),
            ZERO_F64));
  }
  Vectorized<double> map(double (*f)(double)) const {
    float64x2_t result;
    result[0] = f(values[0]);
    result[1] = f(values[1]);
    return result;
  }
  Vectorized<double> map2(
      const Vectorized<double>& second,
      double (*const f)(double, double)) const {
    float64x2_t result;
    result[0] = f(values[0], second.values[0]);
    result[1] = f(values[1], second.values[1]);
    return result;
  }
  Vectorized<double> abs() const {
    return vabsq_f64(values);
  }
  Vectorized<double> angle() const {
    const auto nan_vec = svdup_n_f64(NAN);
    const auto nan_mask =
        svcmpuo_f64(ptrue, svset_neonq(svundef_f64(), values), ZERO_F64);
    const auto pi = svdup_n_f64(c10::pi<double>);

    const auto neg_mask =
        svcmplt_f64(ptrue, svset_neonq(svundef_f64(), values), ZERO_F64);
    auto angle = svsel_f64(neg_mask, pi, ZERO_F64);
    angle = svsel_f64(nan_mask, nan_vec, angle);
    return angle;
  }
  Vectorized<double> real() const {
    return *this;
  }
  Vectorized<double> imag() const {
    return Vectorized<double>(0.0);
  }
  Vectorized<double> conj() const {
    return *this;
  }
  Vectorized<double> acos() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_acosd2_u10(values)), map(std::acos));
  }
  Vectorized<double> acosh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_acoshd2_u10(values)), map(std::acosh));
  }
  Vectorized<double> asin() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_asind2_u10(values)), map(std::asin));
  }
  Vectorized<double> asinh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_asinhd2_u10(values)), map(std::asinh));
  }
  Vectorized<double> atan() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_atand2_u10(values)), map(std::atan));
  }
  Vectorized<double> atanh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_atanhd2_u10(values)), map(std::atanh));
  }
  Vectorized<double> atan2(const Vectorized<double>& b) const {USE_SLEEF(
      { return Vectorized<double>(Sleef_atan2d2_u10(values, b)); },
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); i++) {
          tmp[i] = std::atan2(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      })} Vectorized<double> copysign(const Vectorized<double>& sign) const {
      USE_SLEEF(
          { return Vectorized<double>(Sleef_copysignd2(values, sign)); },
          {
            __at_align__ double tmp[size()];
            __at_align__ double tmp_sign[size()];
            store(tmp);
            sign.store(tmp_sign);
            for (int64_t i = 0; i < size(); i++) {
              tmp[i] = std::copysign(tmp[i], tmp_sign[i]);
            }
            return loadu(tmp);
          })} Vectorized<double> erf() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_erfd2_u10(values)), map(std::erf));
  }
  Vectorized<double> erfc() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_erfcd2_u15(values)), map(std::erfc));
  }
  Vectorized<double> exp() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_expd2_u10(values)), map(std::exp));
  }
  Vectorized<double> exp2() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_exp2d2_u10(values)), map(std::exp2));
  }
  Vectorized<double> expm1() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_expm1d2_u10(values)), map(std::expm1));
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {USE_SLEEF(
      { return Vectorized<double>(Sleef_fmodd2(values, q)); },
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_q[size()];
        store(tmp);
        q.store(tmp_q);
        for (int64_t i = 0; i < size(); i++) {
          tmp[i] = std::fmod(tmp[i], tmp_q[i]);
        }
        return loadu(tmp);
      })} Vectorized<double> hypot(const Vectorized<double>& b) const {
      USE_SLEEF(
          { return Vectorized<double>(Sleef_hypotd2_u05(values, b)); },
          {
            __at_align__ double tmp[size()];
            __at_align__ double tmp_b[size()];
            store(tmp);
            b.store(tmp_b);
            for (int64_t i = 0; i < size(); i++) {
              tmp[i] = std::hypot(tmp[i], tmp_b[i]);
            }
            return loadu(tmp);
          })} Vectorized<double> i0() const {
    return map(calc_i0);
  }
  Vectorized<double> nextafter(const Vectorized<double>& b) const {USE_SLEEF(
      { return Vectorized<double>(Sleef_nextafterd2(values, b)); },
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); ++i) {
          tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      })} Vectorized<double> log() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_logd2_u10(values)), map(std::log));
  }
  Vectorized<double> log2() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_log2d2_u10(values)), map(std::log2));
  }
  Vectorized<double> log10() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_log10d2_u10(values)), map(std::log10));
  }
  Vectorized<double> log1p() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_log1pd2_u10(values)), map(std::log1p));
  }
  Vectorized<double> frac() const;
  Vectorized<double> sin() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_sind2_u10(values)), map(std::sin));
  }
  Vectorized<double> sinh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_sinhd2_u10(values)), map(std::sinh));
  }
  Vectorized<double> cos() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_cosd2_u10(values)), map(std::cos));
  }
  Vectorized<double> cosh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_coshd2_u10(values)), map(std::cosh));
  }
  Vectorized<double> pow(const Vectorized<double>& b) const {USE_SLEEF(
      { return Vectorized<double>(Sleef_powd2_u10(values, b)); },
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); i++) {
          tmp[i] = std::pow(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      })} // Comparison using the _CMP_**_OQ predicate.
          //   `O`: get false if an operand is NaN
          //   `Q`: do not raise if an operand is NaN
  Vectorized<double> tan() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_tand2_u10(values)), map(std::tan));
  }
  Vectorized<double> tanh() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_tanhd2_u10(values)), map(std::tanh));
  }
  Vectorized<double> lgamma() const {
    return USE_SLEEF(
        Vectorized<double>(Sleef_lgammad2_u10(values)), map(std::lgamma));
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> exp_u20() const {
    return exp();
  }
  Vectorized<double> fexp_u20() const {
    return exp();
  }
  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double>& x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> ceil() const {
    return vrndpq_f64(values);
  }
  Vectorized<double> floor() const {
    return vrndmq_f64(values);
  }
  Vectorized<double> neg() const {
    return vnegq_f64(values);
  }
  Vectorized<double> round() const {
    return vrndiq_f64(values);
  }
  Vectorized<double> trunc() const {
    return vrndq_f64(values);
  }
  Vectorized<double> sqrt() const {
    return vsqrtq_f64(values);
  }
  Vectorized<double> reciprocal() const {
    return vdivq_f64(vdupq_n_f64(1.0), values);
  }
  Vectorized<double> rsqrt() const {
    return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(values));
  }
  double reduce_add() const {
    return vaddvq_f64(values);
  }
  double reduce_max() const {
    return vmaxvq_f64(values);
  }
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    return Vectorized<double>(
        vreinterpretq_f64_u64(vceqq_f64(values, other.values)));
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    float64x2_t r0 = vreinterpretq_f64_u32(
        vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(values, other.values))));
    return Vectorized<double>(r0);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    return Vectorized<double>(
        vreinterpretq_f64_u64(vcltq_f64(values, other.values)));
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    return Vectorized<double>(
        vreinterpretq_f64_u64(vcleq_f64(values, other.values)));
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    return Vectorized<double>(
        vreinterpretq_f64_u64(vcgtq_f64(values, other.values)));
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    return Vectorized<double>(
        vreinterpretq_f64_u64(vcgeq_f64(values, other.values)));
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vaddq_f64(a, b);
}

template <>
Vectorized<double> inline operator-(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vsubq_f64(a, b);
}

template <>
Vectorized<double> inline operator*(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vmulq_f64(a, b);
}

template <>
Vectorized<double> inline operator/(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vdivq_f64(a, b);
}

// frac. Implement this here so we can use subtraction
Vectorized<double> inline Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vmaxq_f64(a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vminq_f64(a, b);
}

template <>
Vectorized<double> inline clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min,
    const Vectorized<double>& max) {
  return vminq_f64(max, vmaxq_f64(min, a));
}

template <>
Vectorized<double> inline clamp_max(
    const Vectorized<double>& a,
    const Vectorized<double>& max) {
  return vminq_f64(max, a);
}

template <>
Vectorized<double> inline clamp_min(
    const Vectorized<double>& a,
    const Vectorized<double>& min) {
  return vmaxq_f64(min, a);
}

template <>
Vectorized<double> inline operator&(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(
      vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
Vectorized<double> inline operator|(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(
      vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
Vectorized<double> inline operator^(
    const Vectorized<double>& a,
    const Vectorized<double>& b) {
  return vreinterpretq_f64_u64(
      veorq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

Vectorized<double> inline Vectorized<double>::eq(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this == other), 1));
}

Vectorized<double> inline Vectorized<double>::ne(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this != other), 1));
}

Vectorized<double> inline Vectorized<double>::gt(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this > other), 1));
}

Vectorized<double> inline Vectorized<double>::ge(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this >= other), 1));
}

Vectorized<double> inline Vectorized<double>::lt(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this < other), 1));
}

Vectorized<double> inline Vectorized<double>::le(
    const Vectorized<double>& other) const {
  return svreinterpret_f64_u64(
      svand_n_u64_x(ptrue, svreinterpret_u64_f64(*this <= other), 1));
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<double>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_f64(src);
    auto vec2 = vld1q_f64(src + oneRegElemCount);
    auto vec3 = vld1q_f64(src + twoRegsElemCount);
    auto vec4 = vld1q_f64(src + (twoRegsElemCount + oneRegElemCount));
    vst1q_f64(dst, vec1);
    vst1q_f64(dst + oneRegElemCount, vec2);
    vst1q_f64(dst + twoRegsElemCount, vec3);
    vst1q_f64(dst + (twoRegsElemCount + oneRegElemCount), vec4);
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
Vectorized<double> inline fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return vfmaq_f64(c, a, b);
}

template <>
Vectorized<double> inline fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return vfmsq_f64(c, a, b);
}

template <>
Vectorized<double> inline fmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return svnmsb_f64_x(ptrue, a, b, c);
}

template <>
Vectorized<double> inline fnmsub(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c) {
  return svnmad_f64_x(ptrue, a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
