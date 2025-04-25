#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <cmath>
#if defined(__aarch64__) && defined(AT_BUILD_ARM_VEC256_WITH_SLEEF)
#include <sleef.h>
#define USE_SLEEF(sleef_code, non_sleef_code) sleef_code
#else
#define USE_SLEEF(sleef_code, non_sleef_code) non_sleef_code
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

#if defined(CPU_CAPABILITY_SVE)

template <> class Vectorized<double> {
private:
  vls_float64_t values;
public:
  using value_type = double;
  using size_type = int;
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(double);
  }
  Vectorized() {}
  Vectorized(svfloat64_t v) : values(v) {}
  Vectorized(double val) {
    values = svdup_n_f64(val);
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ double buffer[size()] = { vals... };
    values = svld1_f64(ptrue, buffer);
  }
  operator svfloat64_t() const {
    return values;
  }
  template <uint64_t mask>
  static Vectorized<double> blend(const Vectorized<double>& a, const Vectorized<double>& b) {
    // Build an array of flags: each element is 1 if the corresponding bit in 'mask' is set, 0 otherwise.
    __at_align__ int64_t flag_arr[size()];
    for (int i = 0; i < size(); i++) {
      flag_arr[i] = (mask & (1ULL << i)) ? 1 : 0;
    }
    // Load the flag array into an SVE int64 vector.
    svint64_t int_mask = svld1_s64(svptrue_b64(), flag_arr);
    // Compare each lane of int_mask to 0; returns an svbool_t predicate where true indicates a nonzero flag.
    svbool_t blend_mask = svcmpne_n_s64(svptrue_b64(), int_mask, 0);

    // Use svsel to select elements from b where the predicate is true, else from a.
    svfloat64_t result = svsel(blend_mask, b.values, a.values);
    return Vectorized<double>(result);
  }
  static Vectorized<double> blendv(const Vectorized<double>& a, const Vectorized<double>& b,
                              const Vectorized<double>& mask_) {
    svbool_t mask = svcmpeq_s64(ptrue, svreinterpret_s64_f64(mask_),
                               ALL_S64_TRUE_MASK);
    return svsel_f64(mask, b, a);
  }
  template<typename step_t>
  static Vectorized<double> arange(double base = 0., step_t step = static_cast<step_t>(1)) {
    __at_align__ double buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_f64(ptrue, buffer);
  }
  static Vectorized<double> set(const Vectorized<double>& a, const Vectorized<double>& b,
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
      return svld1_f64(ptrue, reinterpret_cast<const double*>(ptr));
    svbool_t pg = svwhilelt_b64(0ull, count);
    return svld1_f64(pg, reinterpret_cast<const double*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      svst1_f64(ptrue, reinterpret_cast<double*>(ptr), values);
    } else {
      svbool_t pg = svwhilelt_b64(0ull, count);
      svst1_f64(pg, reinterpret_cast<double*>(ptr), values);
    }
  }
  const double& operator[](int idx) const  = delete;
  double& operator[](int idx) = delete;
  int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    __at_align__ int64_t mask_array[size()];

    svbool_t svbool_mask = svcmpeq_f64(ptrue, values, ZERO_F64);
    svst1_s64(ptrue, mask_array, svsel_s64(svbool_mask,
                                          ALL_S64_TRUE_MASK,
                                          ALL_S64_FALSE_MASK));
    for (int64_t i = 0; i < size(); ++i) {
      if (mask_array[i]) mask |= (1ull << i);
    }
    return mask;
  }
  Vectorized<double> isnan() const {
    // NaN check
    svbool_t mask = svcmpuo_f64(ptrue, values, ZERO_F64);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }
  bool has_inf_nan() const {
    return svptest_any(ptrue, svcmpuo_f64(ptrue, svsub_f64_x(ptrue, values, values), ZERO_F64));
  }
  Vectorized<double> map(double (*f)(double)) const {
    __at_align__ double tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> abs() const {
    return svabs_f64_x(ptrue, values);
  }
  Vectorized<double> angle() const {
    const auto nan_vec = svdup_n_f64(NAN);
    const auto nan_mask = svcmpuo_f64(ptrue, values, ZERO_F64);
    const auto pi = svdup_n_f64(c10::pi<double>);

    const auto neg_mask = svcmplt_f64(ptrue, values, ZERO_F64);
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
    return USE_SLEEF(Vectorized<double>(Sleef_acosdx_u10sve(values)),map(std::acos));
  }
  Vectorized<double> acosh() const {
    return USE_SLEEF( Vectorized<double>(Sleef_acoshdx_u10sve(values)),map(std::acosh));
  }
  Vectorized<double> asin() const {
    return USE_SLEEF(Vectorized<double>(Sleef_asindx_u10sve(values)),map(std::asin));
  }
  Vectorized<double> asinh() const {
    return USE_SLEEF(Vectorized<double>(Sleef_asinhdx_u10sve(values)),map(std::asinh));
  }
  Vectorized<double> atan() const {
    return USE_SLEEF(Vectorized<double>(Sleef_atandx_u10sve(values)),map(std::atan));
  }
  Vectorized<double> atanh() const {
    return USE_SLEEF(Vectorized<double>(Sleef_atanhdx_u10sve(values)),map(std::atanh));
  }
  Vectorized<double> atan2(const Vectorized<double> &b) const {
    USE_SLEEF({return Vectorized<double>(Sleef_atan2dx_u10sve(values, b));},
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); i++) {
          tmp[i] = std::atan2(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<double> copysign(const Vectorized<double> &sign) const {
   USE_SLEEF( {return Vectorized<double>(Sleef_copysigndx_sve(values, sign));},
     {
       __at_align__ double tmp[size()];
       __at_align__ double tmp_sign[size()];
       store(tmp);
       sign.store(tmp_sign);
       for (int64_t i = 0; i < size(); i++) {
         tmp[i] = std::copysign(tmp[i], tmp_sign[i]);
       }
       return loadu(tmp);
     }
   )
  }
  Vectorized<double> erf() const {
    return USE_SLEEF(Vectorized<double>(Sleef_erfdx_u10sve(values)),map(std::erf));
  }
  Vectorized<double> erfc() const {
    return USE_SLEEF(Vectorized<double>(Sleef_erfcdx_u15sve(values)),map(std::erfc));
  }
  Vectorized<double> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<double> exp() const {
    return USE_SLEEF(Vectorized<double>(Sleef_expdx_u10sve(values)),map(std::exp));
  }
  Vectorized<double> exp2() const {
    return USE_SLEEF(Vectorized<double>(Sleef_exp2dx_u10sve(values)),map(std::exp2));
  }
  Vectorized<double> expm1() const {
    return USE_SLEEF(Vectorized<double>(Sleef_expm1dx_u10sve(values)),map(std::expm1));
  }
  Vectorized<double> exp_u20() const {
    return exp();
  }
  Vectorized<double> fmod(const Vectorized<double>& q) const {
    USE_SLEEF({return Vectorized<double>(Sleef_fmoddx_sve(values, q));},
    {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_q[size()];
        store(tmp);
        q.store(tmp_q);
        for (int64_t i = 0; i < size(); i++) {
          tmp[i] = std::fmod(tmp[i], tmp_q[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<double> hypot(const Vectorized<double> &b) const {
    USE_SLEEF({return Vectorized<double>(Sleef_hypotdx_u05sve(values, b));},
    {
      __at_align__ double tmp[size()];
      __at_align__ double tmp_b[size()];
      store(tmp);
      b.store(tmp_b);
      for (int64_t i = 0; i < size(); i++) {
        tmp[i] = std::hypot(tmp[i], tmp_b[i]);
      }
      return loadu(tmp);
    })
  }
  Vectorized<double> i0() const {
    return map(calc_i0);
  }
  Vectorized<double> i0e() const {
    return map(calc_i0e);
  }
  Vectorized<double> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<double> igamma(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> igammac(const Vectorized<double> &x) const {
    __at_align__ double tmp[size()];
    __at_align__ double tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<double> nextafter(const Vectorized<double> &b) const {
    USE_SLEEF(
      {
        return Vectorized<double>(Sleef_nextafterdx_sve(values, b));
      },
      {
        __at_align__ double tmp[size()];
        __at_align__ double tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); ++i) {
          tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<double> log() const {
    return USE_SLEEF(Vectorized<double>(Sleef_logdx_u10sve(values)),map(std::log));
  }
  Vectorized<double> log2() const {
    return USE_SLEEF(Vectorized<double>(Sleef_log2dx_u10sve(values)),map(std::log2));
  }
  Vectorized<double> log10() const {
    return USE_SLEEF(Vectorized<double>(Sleef_log10dx_u10sve(values)),map(std::log10));
  }
  Vectorized<double> log1p() const {
    return USE_SLEEF(Vectorized<double>(Sleef_log1pdx_u10sve(values)),map(std::log1p));
  }
  Vectorized<double> frac() const;
  Vectorized<double> sin() const {
    return USE_SLEEF( Vectorized<double>(Sleef_sindx_u10sve(values)),map(std::sin));
  }
  Vectorized<double> sinh() const {
    return USE_SLEEF(Vectorized<double>(Sleef_sinhdx_u10sve(values)),map(std::sinh));
  }
  Vectorized<double> cos() const {
    return USE_SLEEF(Vectorized<double>(Sleef_cosdx_u10sve(values)),map(std::cos));
  }
  Vectorized<double> cosh() const {
    return USE_SLEEF( Vectorized<double>(Sleef_coshdx_u10sve(values)),map(std::cosh));
  }
  Vectorized<double> ceil() const {
    return svrintp_f64_x(ptrue, values);
  }
  Vectorized<double> floor() const {
    return svrintm_f64_x(ptrue, values);
  }
  Vectorized<double> neg() const {
    return svneg_f64_x(ptrue, values);
  }
  Vectorized<double> round() const {
    return svrinti_f64_x(ptrue, values);
  }
  Vectorized<double> tan() const {
    return USE_SLEEF( Vectorized<double>(Sleef_tandx_u10sve(values)),map(std::tan));
  }
  Vectorized<double> tanh() const {
    return USE_SLEEF( Vectorized<double>(Sleef_tanhdx_u10sve(values)),map(std::tanh));
  }
  Vectorized<double> trunc() const {
    return svrintz_f64_x(ptrue, values);
  }
  Vectorized<double> lgamma() const {
    return USE_SLEEF( Vectorized<double>(Sleef_lgammadx_u10sve(values)),map(std::lgamma));
  }
  Vectorized<double> sqrt() const {
    return svsqrt_f64_x(ptrue, values);
  }
  Vectorized<double> reciprocal() const {
    return svdivr_f64_x(ptrue, values, ONE_F64);
  }
  Vectorized<double> rsqrt() const {
    return svdivr_f64_x(ptrue, svsqrt_f64_x(ptrue, values), ONE_F64);
  }
  Vectorized<double> pow(const Vectorized<double> &b) const {
   USE_SLEEF( {return Vectorized<double>(Sleef_powdx_u10sve(values, b));},
    {
      __at_align__ double tmp[size()];
      __at_align__ double tmp_b[size()];
      store(tmp);
      b.store(tmp_b);
      for (int64_t i = 0; i < size(); i++) {
        tmp[i] = std::pow(tmp[i], tmp_b[i]);
      }
      return loadu(tmp);
    }
    )
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<double> operator==(const Vectorized<double>& other) const {
    svbool_t mask = svcmpeq_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator!=(const Vectorized<double>& other) const {
    svbool_t mask = svcmpne_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator<(const Vectorized<double>& other) const {
    svbool_t mask = svcmplt_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator<=(const Vectorized<double>& other) const {
    svbool_t mask = svcmple_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator>(const Vectorized<double>& other) const {
    svbool_t mask = svcmpgt_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> operator>=(const Vectorized<double>& other) const {
    svbool_t mask = svcmpge_f64(ptrue, values, other);
    return svsel_f64(mask, ALL_F64_TRUE_MASK, ALL_F64_FALSE_MASK);
  }

  Vectorized<double> eq(const Vectorized<double>& other) const;
  Vectorized<double> ne(const Vectorized<double>& other) const;
  Vectorized<double> gt(const Vectorized<double>& other) const;
  Vectorized<double> ge(const Vectorized<double>& other) const;
  Vectorized<double> lt(const Vectorized<double>& other) const;
  Vectorized<double> le(const Vectorized<double>& other) const;
};

template <>
Vectorized<double> inline operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svadd_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svsub_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svmul_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svdiv_f64_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
Vectorized<double> inline Vectorized<double>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svmax_f64_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<double> inline minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svmin_f64_x(ptrue, a, b);
}

template <>
Vectorized<double> inline clamp(const Vectorized<double>& a, const Vectorized<double>& min, const Vectorized<double>& max) {
  return svmin_f64_x(ptrue, max, svmax_f64_x(ptrue, min, a));
}

template <>
Vectorized<double> inline clamp_max(const Vectorized<double>& a, const Vectorized<double>& max) {
  return svmin_f64_x(ptrue, max, a);
}

template <>
Vectorized<double> inline clamp_min(const Vectorized<double>& a, const Vectorized<double>& min) {
  return svmax_f64_x(ptrue, min, a);
}

template <>
Vectorized<double> inline operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svreinterpret_f64_s64(svand_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vectorized<double> inline operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svreinterpret_f64_s64(svorr_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

template <>
Vectorized<double> inline operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
  return svreinterpret_f64_s64(sveor_s64_x(ptrue, svreinterpret_s64_f64(a), svreinterpret_s64_f64(b)));
}

Vectorized<double> inline Vectorized<double>::eq(const Vectorized<double>& other) const {
  return (*this == other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::ne(const Vectorized<double>& other) const {
  return (*this != other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::gt(const Vectorized<double>& other) const {
  return (*this > other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::ge(const Vectorized<double>& other) const {
  return (*this >= other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::lt(const Vectorized<double>& other) const {
  return (*this < other) & Vectorized<double>(1.0);
}

Vectorized<double> inline Vectorized<double>::le(const Vectorized<double>& other) const {
  return (*this <= other) & Vectorized<double>(1.0);
}

template <>
inline void convert(const double* src, double* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<double>::size();
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<double>::size()) {
    svst1_f64(ptrue, dst + i, svldnt1_f64(ptrue, src + i));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<double>::size()) {
    svbool_t pg = svwhilelt_b64(i, n);
    svst1_f64(pg, dst + i, svldnt1_f64(pg, src + i));
  }
}

template <>
Vectorized<double> inline fmadd(const Vectorized<double>& a, const Vectorized<double>& b, const Vectorized<double>& c) {
  return svmad_f64_x(ptrue, a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

}}
