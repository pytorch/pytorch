#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <c10/util/irange.h>
#include <cmath>
#if defined(__aarch64__) && defined(AT_BUILD_ARM_VECSVE_WITH_SLEEF)
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

template <> class Vectorized<float> {
private:
  vls_float32_t values;
public:
  using value_type = float;
  using size_type = int;
  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(float);
  }
  Vectorized() {}
  Vectorized(svfloat32_t v) : values(v) {}
  Vectorized(float val) {
    values = svdup_n_f32(val);
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ float buffer[size()] = { vals... };
    values = svld1_f32(ptrue, buffer);
  }
  operator svfloat32_t() const {
    return values;
  }
  static Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask_) {
    svbool_t mask = svcmpeq_s32(ptrue, svreinterpret_s32_f32(mask_),
                               ALL_S32_TRUE_MASK);
    return svsel_f32(mask, b, a);
  }
  template<typename step_t>
  static Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    __at_align__ float buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_f32(ptrue, buffer);
  }
  static Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f32(svwhilelt_b32(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return svld1_f32(ptrue, reinterpret_cast<const float*>(ptr));
    svbool_t pg = svwhilelt_b32(0ull, count);
    return svld1_f32(pg, reinterpret_cast<const float*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      svst1_f32(ptrue, reinterpret_cast<float*>(ptr), values);
    } else {
      svbool_t pg = svwhilelt_b32(0ull, count);
      svst1_f32(pg, reinterpret_cast<float*>(ptr), values);
    }
  }
  const float& operator[](int idx) const  = delete;
  float& operator[](int idx) = delete;
  int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    __at_align__ int32_t mask_array[size()];

    svbool_t svbool_mask = svcmpeq_f32(ptrue, values, ZERO_F32);
    svst1_s32(ptrue, mask_array, svsel_s32(svbool_mask,
                                          ALL_S32_TRUE_MASK,
                                          ALL_S32_FALSE_MASK));
    for (int64_t i = 0; i < size(); ++i) {
      if (mask_array[i]) mask |= (1ull << i);
    }
    return mask;
  }
  Vectorized<float> isnan() const {
    // NaN check
    svbool_t mask = svcmpuo_f32(ptrue, values, ZERO_F32);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  bool has_inf_nan() const {
    return svptest_any(ptrue, svcmpuo_f32(ptrue, svsub_f32_x(ptrue, values, values), ZERO_F32));
  }
  Vectorized<float> map(float (*f)(float)) const {
    __at_align__ float tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> abs() const {
    return svabs_f32_x(ptrue, values);
  }
  Vectorized<float> angle() const {
    const auto nan_vec = svdup_n_f32(NAN);
    const auto nan_mask = svcmpuo_f32(ptrue, values, ZERO_F32);
    const auto pi = svdup_n_f32(c10::pi<float>);

    const auto neg_mask = svcmplt_f32(ptrue, values, ZERO_F32);
    auto angle = svsel_f32(neg_mask, pi, ZERO_F32);
    angle = svsel_f32(nan_mask, nan_vec, angle);
    return angle;
  }
  Vectorized<float> real() const {
    return values;
  }
  Vectorized<float> imag() const {
    return Vectorized<float>(0.f);
  }
  Vectorized<float> conj() const {
    return values;
  }
  Vectorized<float> acos() const {
    return USE_SLEEF(Vectorized<float>(Sleef_acosfx_u10sve(values)),map(std::acos));
  }
  Vectorized<float> acosh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_acoshfx_u10sve(values)),map(std::acosh));
  }
  Vectorized<float> asin() const {
    return USE_SLEEF(Vectorized<float>(Sleef_asinfx_u10sve(values)),map(std::asin));
  }
  Vectorized<float> asinh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_asinhfx_u10sve(values)),map(std::asinh));
  }
  Vectorized<float> atan() const {
    return USE_SLEEF(Vectorized<float>(Sleef_atanfx_u10sve(values)),map(std::atan));
  }
  Vectorized<float> atanh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_atanhfx_u10sve(values)),map(std::atanh));
  }
  Vectorized<float> atan2(const Vectorized<float> &b) const {
     USE_SLEEF({return Vectorized<float>(Sleef_atan2fx_u10sve(values, b));},
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); i++){
          tmp[i] = std::atan2(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
      )
  }
  Vectorized<float> copysign(const Vectorized<float> &sign) const {

    USE_SLEEF({return Vectorized<float>(Sleef_copysignfx_sve(values, sign));},
    {
      __at_align__ float tmp[size()];
      __at_align__ float tmp_sign[size()];
      store(tmp);
      sign.store(tmp_sign);
      for (int64_t i = 0; i < size(); ++i) {
        tmp[i] = std::copysign(tmp[i], tmp_sign[i]);
      }
      return loadu(tmp);
    })
  }
  Vectorized<float> erf() const {
    return USE_SLEEF(Vectorized<float>(Sleef_erffx_u10sve(values)),map(std::erf));
  }
  Vectorized<float> erfc() const {
    return USE_SLEEF(Vectorized<float>(Sleef_erfcfx_u15sve(values)),map(std::erfc));
  }
  Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  Vectorized<float> exp() const {
    return USE_SLEEF(Vectorized<float>(Sleef_expfx_u10sve(values)),map(std::exp));
  }
  Vectorized<float> exp2() const {
    return USE_SLEEF(Vectorized<float>(Sleef_exp2fx_u10sve(values)),map(std::exp2));
  }
  Vectorized<float> expm1() const {
    return USE_SLEEF(Vectorized<float>(Sleef_expm1fx_u10sve(values)),map(std::expm1));
  }
  Vectorized<float> exp_u20() const {
    return exp();
  }
  Vectorized<float> fmod(const Vectorized<float>& q) const {
   USE_SLEEF({return Vectorized<float>(Sleef_fmodfx_sve(values, q));},
    {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_q[size()];
        store(tmp);
        q.store(tmp_q);
        for (int64_t i = 0; i < size(); ++i) {
          tmp[i] = std::fmod(tmp[i], tmp_q[i]);
        }
        return loadu(tmp);
      })
  }
  Vectorized<float> hypot(const Vectorized<float> &b) const {
   USE_SLEEF( {return Vectorized<float>(Sleef_hypotfx_u05sve(values, b));},
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); i++) {
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
  Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  Vectorized<float> igamma(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igamma(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> igammac(const Vectorized<float> &x) const {
    __at_align__ float tmp[size()];
    __at_align__ float tmp_x[size()];
    store(tmp);
    x.store(tmp_x);
    for (int64_t i = 0; i < size(); i++) {
      tmp[i] = calc_igammac(tmp[i], tmp_x[i]);
    }
    return loadu(tmp);
  }
  Vectorized<float> nextafter(const Vectorized<float> &b) const {
    USE_SLEEF(
      {
        return Vectorized<float>(Sleef_nextafterfx_sve(values, b));
      },
      {
        __at_align__ float tmp[size()];
        __at_align__ float tmp_b[size()];
        store(tmp);
        b.store(tmp_b);
        for (int64_t i = 0; i < size(); ++i) {
          tmp[i] = std::nextafter(tmp[i], tmp_b[i]);
        }
        return loadu(tmp);
      }
    )
  }
  Vectorized<float> log() const {
    return USE_SLEEF(Vectorized<float>(Sleef_logfx_u10sve(values)),map(std::log));
  }
  Vectorized<float> log2() const {
    return USE_SLEEF(Vectorized<float>(Sleef_log2fx_u10sve(values)),map(std::log2));
  }
  Vectorized<float> log10() const {
    return USE_SLEEF(Vectorized<float>(Sleef_log10fx_u10sve(values)),map(std::log10));
  }
  Vectorized<float> log1p() const {
    return USE_SLEEF(Vectorized<float>(Sleef_log1pfx_u10sve(values)),map(std::log1p));
  }
  Vectorized<float> frac() const;
  Vectorized<float> sin() const {
    return USE_SLEEF(Vectorized<float>(Sleef_sinfx_u10sve(values)),map(std::sin));
  }
  Vectorized<float> sinh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_sinhfx_u10sve(values)),map(std::sinh));
  }
  Vectorized<float> cos() const {
    return USE_SLEEF(Vectorized<float>(Sleef_cosfx_u10sve(values)),map(std::cos));
  }
  Vectorized<float> cosh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_coshfx_u10sve(values)),map(std::cosh));
  }
  Vectorized<float> ceil() const {
    return svrintp_f32_x(ptrue, values);
  }
  Vectorized<float> floor() const {
    return svrintm_f32_x(ptrue, values);
  }
  Vectorized<float> neg() const {
    return svneg_f32_x(ptrue, values);
  }
  Vectorized<float> round() const {
    return svrinti_f32_x(ptrue, values);
  }
  Vectorized<float> tan() const {
    return USE_SLEEF(Vectorized<float>(Sleef_tanfx_u10sve(values)),map(std::tan));
  }
  Vectorized<float> tanh() const {
    return USE_SLEEF(Vectorized<float>(Sleef_tanhfx_u10sve(values)),map(std::tanh));
  }
  Vectorized<float> trunc() const {
    return svrintz_f32_x(ptrue, values);
  }
  Vectorized<float> lgamma() const {
    return USE_SLEEF(Vectorized<float>(Sleef_lgammafx_u10sve(values)),map(std::lgamma));
  }
  Vectorized<float> sqrt() const {
    return svsqrt_f32_x(ptrue, values);
  }
  Vectorized<float> reciprocal() const {
    return svdivr_f32_x(ptrue, values, ONE_F32);
  }
  Vectorized<float> rsqrt() const {
    return svdivr_f32_x(ptrue, svsqrt_f32_x(ptrue, values), ONE_F32);
  }
  Vectorized<float> pow(const Vectorized<float> &b) const {
   USE_SLEEF( {return Vectorized<float>(Sleef_powfx_u10sve(values, b));},
    {
      __at_align__ float tmp[size()];
      __at_align__ float tmp_b[size()];
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
  Vectorized<float> operator==(const Vectorized<float>& other) const {
    svbool_t mask = svcmpeq_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vectorized<float> operator!=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpne_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vectorized<float> operator<(const Vectorized<float>& other) const {
    svbool_t mask = svcmplt_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vectorized<float> operator<=(const Vectorized<float>& other) const {
    svbool_t mask = svcmple_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vectorized<float> operator>(const Vectorized<float>& other) const {
    svbool_t mask = svcmpgt_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  Vectorized<float> operator>=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpge_f32(ptrue, values, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
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
  return svadd_f32_x(ptrue, a, b);
}

template <>
Vectorized<float> inline operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svsub_f32_x(ptrue, a, b);
}

template <>
Vectorized<float> inline operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmul_f32_x(ptrue, a, b);
}

template <>
Vectorized<float> inline operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svdiv_f32_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
Vectorized<float> inline Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmax_f32_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmin_f32_x(ptrue, a, b);
}

template <>
Vectorized<float> inline clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return svmin_f32_x(ptrue, max, svmax_f32_x(ptrue, min, a));
}

template <>
Vectorized<float> inline clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return svmin_f32_x(ptrue, max, a);
}

template <>
Vectorized<float> inline clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return svmax_f32_x(ptrue, min, a);
}

template <>
Vectorized<float> inline operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svand_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
Vectorized<float> inline operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svorr_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
Vectorized<float> inline operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(sveor_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

Vectorized<float> inline Vectorized<float>::eq(const Vectorized<float>& other) const {
  return (*this == other) & Vectorized<float>(1.0f);
}

Vectorized<float> inline Vectorized<float>::ne(const Vectorized<float>& other) const {
  return (*this != other) & Vectorized<float>(1.0f);
}

Vectorized<float> inline Vectorized<float>::gt(const Vectorized<float>& other) const {
  return (*this > other) & Vectorized<float>(1.0f);
}

Vectorized<float> inline Vectorized<float>::ge(const Vectorized<float>& other) const {
  return (*this >= other) & Vectorized<float>(1.0f);
}

Vectorized<float> inline Vectorized<float>::lt(const Vectorized<float>& other) const {
  return (*this < other) & Vectorized<float>(1.0f);
}

Vectorized<float> inline Vectorized<float>::le(const Vectorized<float>& other) const {
  return (*this <= other) & Vectorized<float>(1.0f);
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<float>::size();
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<float>::size()) {
    svst1_f32(ptrue, dst + i, svldnt1_f32(ptrue, src + i));
  }
  for (int64_t i = n - fraction; i < n; i += Vectorized<float>::size()) {
    svbool_t pg = svwhilelt_b32(i, n);
    svst1_f32(pg, dst + i, svldnt1_f32(pg, src + i));
  }
}

template <>
inline void convert(const float *src, at::Half *dst, int64_t n) {
  const int64_t fraction = n % Vectorized<float>::size();
  svbool_t pg_16 = svwhilelt_b16(0ull, Vectorized<float>::size());
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<float>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<float>::size()) {
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(ptrue, svldnt1_f32(pg_32, src + i)),
                                    ZERO_F16);
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
  for (int64_t i = n - fraction; i < n; i += Vectorized<float>::size()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(ptrue, svldnt1_f32(pg_32, src + i)),
                                     ZERO_F16);
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
}

template <>
inline void convert(const at::Half *src, float *dst, int64_t n) {
  const int64_t fraction = n % Vectorized<float>::size();
  svbool_t pg_16 = svwhilelt_b16(0ull, Vectorized<float>::size());
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<float>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<float>::size()) {
    svfloat16_t src_vec = svzip1_f16(svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
                                    ZERO_F16);
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(ptrue, src_vec));
  }
  for (int64_t i =  n - fraction; i < n; i += Vectorized<float>::size()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svzip1_f16(svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
                                     ZERO_F16);
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(ptrue, src_vec));
  }
}

template <>
inline void convert(const bool *src, float *dst, int64_t n) {
  const int64_t fraction = n % Vectorized<float>::size();
  svbool_t pg_8 = svwhilelt_b8(0ull, Vectorized<float>::size());
  svbool_t pg_32 = svwhilelt_b32(0ull, Vectorized<float>::size());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<float>::size()) {
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_f32(pg_32, dst + i, svsel_f32(mask, ONE_F32, ZERO_F32));
  }
  for (int64_t i = n - fraction; i < n; i += Vectorized<float>::size()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_f32(pg_32, dst + i, svsel_f32(mask, ONE_F32, ZERO_F32));
  }
}

template <>
Vectorized<float> inline fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return svmad_f32_x(ptrue, a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

}}
