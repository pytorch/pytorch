#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <ATen/cpu/vec/sve/sve_helper.h>

#include <algorithm>
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

#if defined(CPU_CAPABILITY_SVE) || defined(CPU_CAPABILITY_SVE256)

template <> class Vectorized<float> {
private:

public:
  __at_align__ float values[64];

  using value_type = float;
  using size_type = int;
  static inline size_type size() {
    return svcntw();
  }
  inline Vectorized() {}
  inline Vectorized(const float val) {
    svst1_f32(svptrue_b32(), values, svdup_n_f32(val));
  }
  inline Vectorized(const svfloat32_t val) {
    svst1_f32(svptrue_b32(), values, val);
  }
  template<typename T,
           typename = std::enable_if_t<std::is_pointer_v<T>>>
  inline Vectorized(const float * val) {
    svst1_f32(svptrue_b32(), values, svld1_f32(svptrue_b32(), val));
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  inline Vectorized(Args... vals) {
    values = { vals... };
  }
  inline operator svfloat32_t() const {
    return svld1_f32(svptrue_b32(), values);
  }
  static inline Vectorized<float> from_ptr(const float * vs) {
    Vectorized<float> v;
    svst1_f32(svptrue_b32(), v.values, svld1_f32(svptrue_b32(), static_cast<const float *>(vs)));
    return v;
  }
  static inline Vectorized<float> from_ptr(const float * vs, int count) {
    Vectorized<float> v;
    svst1_f32(svptrue_b32(), v.values, svld1_f32(svwhilelt_b32_s32(0, count), static_cast<const float *>(vs)));
    return v;
  }
  inline void set_lane(int i, float value) {
    values[i] = value;
  }
  inline Vectorized<float> map(float (*fn)(float)) const {
    Vectorized<float> result;
    for (int64_t i = 0; i < size(); ++i) {
      result.set_lane(i, fn(values[i]));
    }
    return result;
  }
  inline Vectorized<float> map2(float (*fn)(float, float), const Vectorized<float> &b) const {
    Vectorized<float> result;
    for (int64_t i = 0; i < size(); ++i) {
      result.set_lane(i, fn(values[i], b.values[i]));
    }
    return result;
  }

  static inline Vectorized<float> blend(const Vectorized<float>& a, const Vectorized<float>& b, const uint64_t mask) {
    // Build an array of flags: each element is 1 if the corresponding bit in 'mask' is set, 0 otherwise.
    __at_align__ int32_t flag_arr[size()];
    for (int i = 0; i < size(); i++) {
        flag_arr[i] = (mask & (1ULL << i)) ? 1 : 0;
    }
    // Load the flag array into an SVE int32 vector.
    svint32_t int_mask = svld1_s32(svptrue_b32(), flag_arr);
    // Compare each lane of int_mask to 0; returns an svbool_t predicate where true indicates a nonzero flag.
    svbool_t blend_mask = svcmpne_n_s32(svptrue_b32(), int_mask, 0);
    // Use svsel to select elements from b where the predicate is true, else from a.
    return svsel_f32(blend_mask, b, a);
  }
  static inline Vectorized<float> blendv(const Vectorized<float>& a, const Vectorized<float>& b,
                              const Vectorized<float>& mask_) {
    svbool_t mask = svcmpeq_s32(svptrue_b32(), svreinterpret_s32_f32(mask_), ALL_S32_TRUE_MASK);
    return svsel_f32(mask, b, a);
  }
  template<typename step_t>
  static inline Vectorized<float> arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
    __at_align__ float buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return Vectorized<float>::from_ptr(buffer);
  }
  static inline Vectorized<float> set(const Vectorized<float>& a, const Vectorized<float>& b,
                           int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f32(svwhilelt_b32(0ull, count), b, a);
    }
    return b;
  }
  static inline Vectorized<float> loadu(const void* ptr) {
    return Vectorized<float>::from_ptr(reinterpret_cast<const float *>(ptr));
  }
  static inline Vectorized<float> loadu(const void* ptr, int64_t count) {
    return Vectorized<float>::from_ptr(reinterpret_cast<const float *>(ptr), count);
  }
  inline void store(void* ptr) const {
    svst1_f32(svptrue_b32(), static_cast<float *>(ptr), svld1_f32(svptrue_b32(), values));
  }
  inline void store(void* ptr, int count) const {
    svst1_f32(svwhilelt_b32_s32(0, count), static_cast<float *>(ptr), svld1_f32(svptrue_b32(), values));
  }
  inline const float& operator[](int idx) const {
    return values[idx];
  };
  inline float& operator[](int idx) {
    return values[idx];
  };
  inline int64_t zero_mask() const {
    // returns an integer mask where all zero elements are translated to 1-bit and others are translated to 0-bit
    int64_t mask = 0;
    __at_align__ int32_t mask_array[size()];

    svbool_t svbool_mask = svcmpeq_f32(svptrue_b32(), *this, ZERO_F32);
    svst1_s32(svptrue_b32(), mask_array, svsel_s32(svbool_mask,
                                          ALL_S32_TRUE_MASK,
                                          ALL_S32_FALSE_MASK));
    for (int64_t j = 0; j < size(); ++j) {
      if (mask_array[j]) mask |= (1ull << j);
    }

    return mask;
  }
  inline Vectorized<float> isnan() const {
    // NaN check
    auto mask = svcmpuo_f32(svptrue_b32(), *this, ZERO_F32);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline bool has_inf_nan() const {
    return svptest_any(svptrue_b32(), svcmpuo_f32(svptrue_b32(), svsub_f32_x(svptrue_b32(), *this, *this), ZERO_F32));
  }
  
  inline Vectorized<float> abs() const {
    return svabs_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> angle() const {
    const auto nan_vec = svdup_n_f32(NAN);
    const auto nan_mask = svcmpuo_f32(svptrue_b32(), *this, ZERO_F32);
    const auto pi = svdup_n_f32(c10::pi<float>);
    const auto neg_mask = svcmplt_f32(svptrue_b32(), *this, ZERO_F32);
    auto angle = svsel_f32(neg_mask, pi, ZERO_F32);
    return svsel_f32(nan_mask, nan_vec, angle);
  }
  inline Vectorized<float> real() const {
    return *this;
  }
  inline Vectorized<float> imag() const {
    return Vectorized<float>(0.f);
  }
  inline Vectorized<float> conj() const {
    return *this;
  }
  inline Vectorized<float> acos() const {
    return USE_SLEEF(Sleef_acosfx_u10sve(*this), map(std::acos));
  }
  inline Vectorized<float> acosh() const {
    return USE_SLEEF(Sleef_acoshfx_u10sve(*this), map(std::acosh));
  }
  inline Vectorized<float> asin() const {
    return USE_SLEEF(Sleef_asinfx_u10sve(*this), map(std::asin));
  }
  inline Vectorized<float> asinh() const {
    return USE_SLEEF(Sleef_asinhfx_u10sve(*this), map(std::asinh));
  }
  inline Vectorized<float> atan() const {
    return USE_SLEEF(Sleef_atanfx_u10sve(*this), map(std::atan));
  }
  inline Vectorized<float> atanh() const {
    return USE_SLEEF(Sleef_atanhfx_u10sve(*this), map(std::atanh));
  }
  inline Vectorized<float> atan2(const Vectorized<float> &b) const {
    return USE_SLEEF(Sleef_atan2fx_u10sve(*this, b), map2(std::atan2, b));
  }
  inline Vectorized<float> copysign(const Vectorized<float> &sign) const {
    return USE_SLEEF(Sleef_copysignfx_sve(*this, sign), map2(std::copysign, sign));
  }
  inline Vectorized<float> erf() const {
    return USE_SLEEF(Sleef_erffx_u10sve(*this), map(std::erf));
  }
  inline Vectorized<float> erfc() const {
    return USE_SLEEF(Sleef_erfcfx_u15sve(*this), map(std::erfc));
  }
  inline Vectorized<float> erfinv() const {
    return map(calc_erfinv);
  }
  inline Vectorized<float> exp() const {
    return USE_SLEEF(Sleef_expfx_u10sve(*this), map(std::exp));
  }
  inline Vectorized<float> exp2() const {
    return USE_SLEEF(Sleef_exp2fx_u10sve(*this), map(std::exp2));
  }
  inline Vectorized<float> expm1() const {
    return USE_SLEEF(Sleef_expm1fx_u10sve(*this), map(std::expm1));
  }
  inline Vectorized<float> exp_u20() {
    return exp();
  }
  inline Vectorized<float> fmod(const Vectorized<float>& q) const {
    return USE_SLEEF(Sleef_fmodfx_sve(*this, q), return map2(std::fmod, q));
  }
  inline Vectorized<float> hypot(const Vectorized<float> &b) const {
   return USE_SLEEF(Sleef_hypotfx_u05sve(*this, b), map2(std::hypot, b));
  }
  inline Vectorized<float> i0() const {
    return map(calc_i0);
  }
  inline Vectorized<float> i0e() const {
    return map(calc_i0e<float>);
  }
  inline Vectorized<float> digamma() const {
    return map(calc_digamma);
  }
  inline Vectorized<float> igamma(const Vectorized<float> &x) const {
    return map2(calc_igamma<float>, x);
  }
  inline Vectorized<float> igammac(const Vectorized<float> &x) const {
    return map2(calc_igammac<float>, x);
  }
  inline Vectorized<float> nextafter(const Vectorized<float> &b) const {
    return USE_SLEEF(Sleef_nextafterfx_sve(*this, b), map2(std::nextafter, b));
  }
  inline Vectorized<float> log() const {
    return USE_SLEEF(Sleef_logfx_u10sve(*this), map(std::log));
  }
  inline Vectorized<float> log2() const {
    return USE_SLEEF(Sleef_log2fx_u10sve(*this), map(std::log2));
  }
  inline Vectorized<float> log10() const {
    return USE_SLEEF(Sleef_log10fx_u10sve(*this), map(std::log10));
  }
  inline Vectorized<float> log1p() const {
    return USE_SLEEF(Sleef_log1pfx_u10sve(*this), map(std::log1p));
  }
  inline Vectorized<float> frac() const;
  inline Vectorized<float> sin() const {
    return USE_SLEEF(Sleef_sinfx_u10sve(*this), map(std::sin));
  }
  inline Vectorized<float> sinh() const {
    return USE_SLEEF(Sleef_sinhfx_u10sve(*this), map(std::sinh));
  }
  inline Vectorized<float> cos() const {
    return USE_SLEEF(Sleef_cosfx_u10sve(*this), map(std::cos));
  }
  inline Vectorized<float> cosh() const {
    return USE_SLEEF(Sleef_coshfx_u10sve(*this), map(std::cosh));
  }
  inline Vectorized<float> ceil() const {
    return svrintp_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> floor() const {
    return svrintm_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> neg() const {
    return svneg_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> round() const {
    return svrinti_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> tan() const {
    return USE_SLEEF(Sleef_tanfx_u10sve(*this), map(std::tan));
  }
  inline Vectorized<float> tanh() const {
    return USE_SLEEF(Sleef_tanhfx_u10sve(*this), map(std::tanh));
  }
  inline Vectorized<float> trunc() const {
    return svrintz_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> lgamma() const {
    return USE_SLEEF(Sleef_lgammafx_u10sve(*this), map(std::lgamma));
  }
  inline Vectorized<float> sqrt() const {
    return svsqrt_f32_x(svptrue_b32(), *this);
  }
  inline Vectorized<float> reciprocal() const {
    return svdivr_f32_x(svptrue_b32(), *this, svdup_n_f32(1.f));
  }
  inline Vectorized<float> rsqrt() const {
    return svdivr_f32_x(svptrue_b32(), svsqrt_f32_x(svptrue_b32(), *this), ONE_F32);
  }
  inline Vectorized<float> pow(const Vectorized<float> &b) const {
    return USE_SLEEF(Sleef_powfx_u10sve(*this, b), map(std::pow, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  inline Vectorized<float> operator==(const Vectorized<float>& other) const {
    svbool_t mask = svcmpeq_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline Vectorized<float> operator!=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpne_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline Vectorized<float> operator<(const Vectorized<float>& other) const {
    svbool_t mask = svcmplt_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator<=(const Vectorized<float>& other) const {
    svbool_t mask = svcmple_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator>(const Vectorized<float>& other) const {
    svbool_t mask = svcmpgt_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator>=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpge_f32(svptrue_b32(), *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> eq(const Vectorized<float>& other) const;
  inline Vectorized<float> ne(const Vectorized<float>& other) const;
  inline Vectorized<float> gt(const Vectorized<float>& other) const;
  inline Vectorized<float> ge(const Vectorized<float>& other) const;
  inline Vectorized<float> lt(const Vectorized<float>& other) const;
  inline Vectorized<float> le(const Vectorized<float>& other) const;
};

template <>
inline Vectorized<float> operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svadd_f32_x(svptrue_b32(), a, b);
}

template <>
inline Vectorized<float> operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svsub_f32_x(svptrue_b32(), a, b);
}

template <>
inline Vectorized<float> operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmul_f32_x(svptrue_b32(), a, b);
}

template <>
inline Vectorized<float> operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svdiv_f32_x(svptrue_b32(), a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmax_f32_x(svptrue_b32(), a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmin_f32_x(svptrue_b32(), a, b);
}

template <>
inline Vectorized<float> clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return svmin_f32_x(svptrue_b32(), max, svmax_f32_x(svptrue_b32(), min, a));
}

template <>
inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return svmin_f32_x(svptrue_b32(), max, a);
}

template <>
inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return svmax_f32_x(svptrue_b32(), min, a);
}

template <>
inline Vectorized<float> operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svand_s32_x(svptrue_b32(), svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
inline Vectorized<float> operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svorr_s32_x(svptrue_b32(), svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
inline Vectorized<float> operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(sveor_s32_x(svptrue_b32(), svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
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
inline void convert(const float* src, float* dst, int64_t n) {
  const int64_t fraction = n % svcntw();
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += svcntw()) {
    svst1_f32(svptrue_b32(), dst + i, svldnt1_f32(svptrue_b32(), src + i));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += svcntw()) {
    svbool_t pg = svwhilelt_b32(i, n);
    svst1_f32(pg, dst + i, svldnt1_f32(pg, src + i));
  }
}

template <>
inline void convert(const float *src, at::Half *dst, int64_t n) {
  const int64_t fraction = n % svcntw();
  svbool_t pg_16 = svwhilelt_b16(0ull, svcntw());
  svbool_t pg_32 = svwhilelt_b32(0ull, svcntw());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += svcntw()) {
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(svptrue_b32(), svldnt1_f32(pg_32, src + i)),
                                    ZERO_F16);
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += svcntw()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(svptrue_b32(), svldnt1_f32(pg_32, src + i)),
                                     ZERO_F16);
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
}

template <>
inline void convert(const at::Half *src, float *dst, int64_t n) {
  const int64_t fraction = n % svcntw();
  svbool_t pg_16 = svwhilelt_b16(0ull, svcntw());
  svbool_t pg_32 = svwhilelt_b32(0ull, svcntw());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += svcntw()) {
    svfloat16_t src_vec = svzip1_f16(svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
                                    ZERO_F16);
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(svptrue_b32(), src_vec));
  }
#pragma unroll
  for (int64_t i =  n - fraction; i < n; i += svcntw()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svzip1_f16(svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
                                     ZERO_F16);
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(svptrue_b32(), src_vec));
  }
}

template <>
inline void convert(const bool *src, float *dst, int64_t n) {
  const int64_t fraction = n % svcntw();
  svbool_t pg_8 = svwhilelt_b8(0ull, svcntw());
  svbool_t pg_32 = svwhilelt_b32(0ull, svcntw());
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += svcntw()) {
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_f32(pg_32, dst + i, svsel_f32(mask, ONE_F32, ZERO_F32));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += svcntw()) {
    pg_8 = svwhilelt_b8(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svuint8_t src_vec_u8 = svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_f32(pg_32, dst + i, svsel_f32(mask, ONE_F32, ZERO_F32));
  }
}

template <>
inline Vectorized<float> fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return svmad_f32_x(svptrue_b32(), a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

}}
