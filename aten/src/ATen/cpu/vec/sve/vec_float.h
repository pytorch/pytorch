#pragma once

#include <ATen/cpu/vec/intrinsics.h>
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

template <>
struct is_vec_specialized_for<float> : std::bool_constant<true> {};

template <>
class Vectorized<float> {
 private:
    __at_align__ float values[2048 / sizeof(float)];
 public:

  using value_type = float;
  using size_type = int;
  static inline size_type size() {
    return svcntw();
  }
  inline Vectorized() {svst1_f32(ptrue, values, svdup_n_f32(0));}
  inline Vectorized(const float val) {
    svst1_f32(ptrue, values, svdup_n_f32(val));
  }
  inline Vectorized(const svfloat32_t val) {
    svst1_f32(ptrue, values, val);
  }
  template<typename T,
           typename = std::enable_if_t<std::is_pointer_v<T>>>
  inline Vectorized(float * val) {
    svst1_f32(ptrue, values, svld1_f32(ptrue, val));
  }
  template<typename... Args,
           typename = std::enable_if_t<(sizeof...(Args) == size())>>
  inline Vectorized(Args... vals) {
    values = { vals... };
  }
  inline operator svfloat32_t() const {
    return svld1_f32(ptrue, values);
  }
  static inline Vectorized<float> from_ptr(const float * vs) {
    Vectorized<float> v;
    svst1_f32(ptrue, v.values, svld1_f32(ptrue, static_cast<const float *>(vs)));
    return v;
  }
  static inline Vectorized<float> from_ptr(const float * vs, int count) {
    Vectorized<float> v;
    svst1_f32(ptrue, v.values, svld1_f32(svwhilelt_b32_s32(0, count), static_cast<const float *>(vs)));
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
    __at_align__ int32_t * flag_arr = new int32_t[size()];
    for (int i = 0; i < size(); i++) {
      flag_arr[i] = (mask & (1ULL << i)) ? 1 : 0;
    }
    // Load the flag array into an SVE int32 vector.
    svint32_t int_mask = svld1_s32(ptrue, flag_arr);
    delete[] flag_arr;
    // Compare each lane of int_mask to 0; returns an svbool_t predicate where true indicates a nonzero flag.
    svbool_t blend_mask = svcmpne_n_s32(ptrue, int_mask, 0);
    // Use svsel to select elements from b where the predicate is true, else from a.
    return svsel_f32(blend_mask, b, a);
  }
  static inline Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask_) {
    svbool_t mask =
        svcmpeq_s32(ptrue, svreinterpret_s32_f32(mask_), ALL_S32_TRUE_MASK);
    return svsel_f32(mask, b, a);
  }
  template <typename step_t>
  static inline Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    __at_align__ float * buffer = new float[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    auto tmp = Vectorized<float>::from_ptr(buffer);
    delete[] buffer;
    return tmp;
  }
  static inline Vectorized<float> set(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_f32(svwhilelt_b32(0ull, count), b, a);
    }
    return b;
  }
  // Implementation is picked from
  // https://github.com/ARM-software/ComputeLibrary/blob/v25.01/src/core/NEON/SVEMath.inl#L105
  inline svfloat32_t svexp_f32_z(svbool_t pg, svfloat32_t x) const {
    const auto c1 =
        svreinterpret_f32_u32(svdup_n_u32(0x3f7ffff6)); // x^1: 0x1.ffffecp-1f
    const auto c2 =
        svreinterpret_f32_u32(svdup_n_u32(0x3efffedb)); // x^2: 0x1.fffdb6p-2f
    const auto c3 =
        svreinterpret_f32_u32(svdup_n_u32(0x3e2aaf33)); // x^3: 0x1.555e66p-3f
    const auto c4 =
        svreinterpret_f32_u32(svdup_n_u32(0x3d2b9f17)); // x^4: 0x1.573e2ep-5f
    const auto c5 =
        svreinterpret_f32_u32(svdup_n_u32(0x3c072010)); // x^5: 0x1.0e4020p-7f
    const auto shift = svreinterpret_f32_u32(
        svdup_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = svreinterpret_f32_u32(
        svdup_n_u32(0x3fb8aa3b)); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi = svreinterpret_f32_u32(svdup_n_u32(
        0xbf317200)); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo = svreinterpret_f32_u32(svdup_n_u32(
        0xb5bfbe8e)); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f
    const auto inf = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f); // Approximately ln(2^127.5)
    const auto zero = svdup_n_f32(0.f);
    const auto min_input = svdup_n_f32(-86.64f); // Approximately ln(2^-125)
    // Range reduction:
    //   e^x = 2^n * e^r
    // where:
    //   n = floor(x / ln(2))
    //   r = x - n * ln(2)
    //
    // By adding x / ln(2) with 2^23 + 127 (shift):
    //   * As FP32 fraction part only has 23-bits, the addition of 2^23 + 127
    //   forces decimal part
    //     of x / ln(2) out of the result. The integer part of x / ln(2) (i.e.
    //     n) + 127 will occupy the whole fraction part of z in FP32 format.
    //     Subtracting 2^23 + 127 (shift) from z will result in the integer part
    //     of x / ln(2) (i.e. n) because the decimal part has been pushed out
    //     and lost.
    //   * The addition of 127 makes the FP32 fraction part of z ready to be
    //   used as the exponent
    //     in FP32 format. Left shifting z by 23 bits will result in 2^n.
    const auto z = svmla_f32_z(pg, shift, x, inv_ln2);
    const auto n = svsub_f32_z(pg, z, shift);
    const auto scale = svreinterpret_f32_u32(
        svlsl_n_u32_z(pg, svreinterpret_u32_f32(z), 23)); // 2^n
    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy
    // beyond FP32. This outperforms longer Taylor series (3-4 tabs) both in
    // term of accuracy and performance.
    const auto r_hi = svmla_f32_z(pg, x, n, neg_ln2_hi);
    const auto r = svmla_f32_z(pg, r_hi, n, neg_ln2_lo);
    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = svmul_f32_z(pg, r, r);
    const auto p1 = svmul_f32_z(pg, c1, r);
    const auto p23 = svmla_f32_z(pg, c2, c3, r);
    const auto p45 = svmla_f32_z(pg, c4, c5, r);
    const auto p2345 = svmla_f32_z(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_z(pg, p1, p2345, r2);
    auto poly = svmla_f32_z(pg, scale, p12345, scale);
    // Handle underflow and overflow.
    poly = svsel_f32(svcmplt_f32(pg, x, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, x, max_input), inf, poly);
    return poly;
  }
  static inline Vectorized<float> loadu(const void* ptr) {
    return Vectorized<float>::from_ptr(reinterpret_cast<const float *>(ptr));
  }
  static inline Vectorized<float> loadu(const void* ptr, int64_t count) {
    return Vectorized<float>::from_ptr(reinterpret_cast<const float *>(ptr), count);
  }
  inline void store(void* ptr) const {
    svst1_f32(ptrue, static_cast<float *>(ptr), svld1_f32(ptrue, values));
  }
  inline void store(void* ptr, int count) const {
    svst1_f32(svwhilelt_b32_s32(0, count), static_cast<float *>(ptr), svld1_f32(ptrue, values));
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
    __at_align__ int32_t * mask_array = new int32_t[size()];

    svbool_t svbool_mask = svcmpeq_f32(ptrue, *this, ZERO_F32);
    svst1_s32(ptrue, mask_array, svsel_s32(svbool_mask,
                                          ALL_S32_TRUE_MASK,
                                          ALL_S32_FALSE_MASK));
    for (int64_t j = 0; j < size(); ++j) {
      if (mask_array[j]) mask |= (1ull << j);
    }
    delete[] mask_array;
    return mask;
  }
  inline Vectorized<float> isnan() const {
    // NaN check
    auto mask = svcmpuo_f32(ptrue, *this, ZERO_F32);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline bool has_inf_nan() const {
    return svptest_any(ptrue, svcmpuo_f32(ptrue, svsub_f32_x(ptrue, *this, *this), ZERO_F32));
  }
  
  inline Vectorized<float> abs() const {
    return svabs_f32_x(ptrue, *this);
  }
  inline Vectorized<float> angle() const {
    const auto nan_vec = svdup_n_f32(NAN);
    const auto nan_mask = svcmpuo_f32(ptrue, *this, ZERO_F32);
    const auto pi = svdup_n_f32(c10::pi<float>);
    const auto neg_mask = svcmplt_f32(ptrue, *this, ZERO_F32);
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
  // Implementation copied from Arm Optimized Routines: 
  // https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/sve/expf.c
  Vectorized<float> exp_u20() const {
    // special case to handle special inputs that are too large or too small
    // i.e. where there's at least one element x, s.t. |x| >= 87.3...
    svbool_t is_special_case = svacgt (svptrue_b32(), *this, 0x1.5d5e2ap+6f);
    if (svptest_any (svptrue_b32(), is_special_case)) {
      return exp();
    }
    const svfloat32_t ln2_hi = svdup_n_f32(0x1.62e4p-1f);    
    const svfloat32_t ln2_lo = svdup_n_f32(0x1.7f7d1cp-20f);    
    const svfloat32_t c1 = svdup_n_f32(0.5f);    
    const svfloat32_t inv_ln2 = svdup_n_f32(0x1.715476p+0f);

    const float shift = 0x1.803f8p17f;    

    /* n = round(x/(ln2/N)).  */
    svfloat32_t z = svmad_x (svptrue_b32(), inv_ln2, *this, shift);
    svfloat32_t n = svsub_x (svptrue_b32(), z, shift);

    /* r = x - n*ln2/N.  */
    svfloat32_t r = *this;
    r = svmls_x(svptrue_b32(), r, n, ln2_hi);
    r = svmls_x(svptrue_b32(), r, n, ln2_lo);

    /* scale = 2^(n/N).  */
    svfloat32_t scale = svexpa (svreinterpret_u32 (z));

    /* poly(r) = exp(r) - 1 ~= r + 0.5 r^2.  */
    svfloat32_t r2 = svmul_x (svptrue_b32 (), r, r);
    svfloat32_t poly = svmla_x(svptrue_b32(), r, r2, c1);
    return svmla_x (svptrue_b32(), scale, scale, poly);
  }
  Vectorized<float> fexp_u20() const {
    return exp_u20();
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
    return svrintp_f32_x(ptrue, *this);
  }
  inline Vectorized<float> floor() const {
    return svrintm_f32_x(ptrue, *this);
  }
  inline Vectorized<float> neg() const {
    return svneg_f32_x(ptrue, *this);
  }
  inline Vectorized<float> round() const {
    return svrinti_f32_x(ptrue, *this);
  }
  inline Vectorized<float> tan() const {
    return USE_SLEEF(Sleef_tanfx_u10sve(*this), map(std::tan));
  }
  // Implementation is picked from
  // https://github.com/ARM-software/ComputeLibrary/blob/v25.01/src/core/NEON/SVEMath.inl#L179
  inline Vectorized<float> tanh() const {
    // Constants used for the tanh calculation.
    const svfloat32_t CONST_1 =
        svdup_n_f32(1.f); // Constant 1.0f for the tanh formula.
    const svfloat32_t CONST_2 = svdup_n_f32(
        2.f); // Constant 2.0f for the tanh formula (used in exp(2x)).
    const svfloat32_t CONST_MIN_TANH = svdup_n_f32(
        -10.f); // Minimum threshold for input values to prevent overflow.
    const svfloat32_t CONST_MAX_TANH = svdup_n_f32(
        10.f); // Maximum threshold for input values to prevent overflow.

    // Step 1: Clamp the values within the range [-10, 10] to prevent overflow
    // during exponentiation. The tanh function approaches ±1 rapidly as the
    // input grows large, so we limit the input range to avoid numerical
    // instability. svmax_f32_z ensures values are greater than -10, and
    // svmin_f32_z ensures they are less than 10.
    svfloat32_t x = svmin_f32_z(
        ptrue, svmax_f32_z(ptrue, *this, CONST_MIN_TANH), CONST_MAX_TANH);

    // Step 2: Calculate exp(2 * x), where x is the clamped value.
    // svmul_f32_z computes 2 * x, and svexp_f32_z computes the exponential of
    // the result.
    svfloat32_t exp2x = svexp_f32_z(ptrue, svmul_f32_z(ptrue, CONST_2, x));

    // Step 3: Calculate the numerator of the tanh function, which is exp(2x)
    // - 1.
    svfloat32_t num = svsub_f32_z(ptrue, exp2x, CONST_1);

    // Step 4: Calculate the denominator of the tanh function, which is exp(2x)
    // + 1.
    svfloat32_t den = svadd_f32_z(ptrue, exp2x, CONST_1);

    // Step 5: Calculate the tanh function as the ratio of the numerator and
    // denominator: num / den.
    svfloat32_t tanh = svdiv_f32_z(ptrue, num, den);

    // Return the calculated tanh values.
    return tanh;
  }
  inline Vectorized<float> trunc() const {
    return svrintz_f32_x(ptrue, *this);
  }
  inline Vectorized<float> lgamma() const {
    return USE_SLEEF(Sleef_lgammafx_u10sve(*this), map(std::lgamma));
  }
  inline Vectorized<float> sqrt() const {
    return svsqrt_f32_x(ptrue, *this);
  }
  inline Vectorized<float> reciprocal() const {
    return svdivr_f32_x(ptrue, *this, svdup_n_f32(1.f));
  }
  inline Vectorized<float> rsqrt() const {
    return svdivr_f32_x(ptrue, svsqrt_f32_x(ptrue, *this), ONE_F32);
  }
  inline Vectorized<float> pow(const Vectorized<float> &b) const {
    return USE_SLEEF(Sleef_powfx_u10sve(*this, b), map(std::pow, b));
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  inline Vectorized<float> operator==(const Vectorized<float>& other) const {
    svbool_t mask = svcmpeq_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline Vectorized<float> operator!=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpne_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }
  inline Vectorized<float> operator<(const Vectorized<float>& other) const {
    svbool_t mask = svcmplt_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator<=(const Vectorized<float>& other) const {
    svbool_t mask = svcmple_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator>(const Vectorized<float>& other) const {
    svbool_t mask = svcmpgt_f32(ptrue, *this, other);
    return svsel_f32(mask, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  }

  inline Vectorized<float> operator>=(const Vectorized<float>& other) const {
    svbool_t mask = svcmpge_f32(ptrue, *this, other);
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
  return svadd_f32_x(ptrue, a, b);
}

template <>
inline Vectorized<float> operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svsub_f32_x(ptrue, a, b);
}

template <>
inline Vectorized<float> operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmul_f32_x(ptrue, a, b);
}

template <>
inline Vectorized<float> operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svdiv_f32_x(ptrue, a, b);
}

// frac. Implement this here so we can use subtraction
inline Vectorized<float> Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<float> inline maximum(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  return svmax_f32_x(ptrue, a, b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svmin_f32_x(ptrue, a, b);
}

template <>
inline Vectorized<float> clamp(const Vectorized<float>& a, const Vectorized<float>& min, const Vectorized<float>& max) {
  return svmin_f32_x(ptrue, max, svmax_f32_x(ptrue, min, a));
}

template <>
inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max) {
  return svmin_f32_x(ptrue, max, a);
}

template <>
inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min) {
  return svmax_f32_x(ptrue, min, a);
}

template <>
inline Vectorized<float> operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svand_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
inline Vectorized<float> operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(svorr_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
}

template <>
inline Vectorized<float> operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
  return svreinterpret_f32_s32(sveor_s32_x(ptrue, svreinterpret_s32_f32(a), svreinterpret_s32_f32(b)));
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
    svst1_f32(ptrue, dst + i, svldnt1_f32(ptrue, src + i));
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
    svfloat16_t src_vec = svuzp1_f16(svcvt_f16_f32_x(ptrue, svldnt1_f32(pg_32, src + i)),
                                    ZERO_F16);
    svst1_f16(pg_16, reinterpret_cast<float16_t*>(dst) + i, src_vec);
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += svcntw()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svuzp1_f16(
        svcvt_f16_f32_x(ptrue, svldnt1_f32(pg_32, src + i)), ZERO_F16);
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
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(ptrue, src_vec));
  }
#pragma unroll
  for (int64_t i =  n - fraction; i < n; i += svcntw()) {
    pg_16 = svwhilelt_b16(i, n);
    pg_32 = svwhilelt_b32(i, n);
    svfloat16_t src_vec = svzip1_f16(
        svldnt1_f16(pg_16, reinterpret_cast<const float16_t*>(src) + i),
        ZERO_F16);
    svst1_f32(pg_32, dst + i, svcvt_f32_f16_x(ptrue, src_vec));
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
    svuint8_t src_vec_u8 =
        svldnt1_u8(pg_8, reinterpret_cast<const uint8_t*>(src) + i);
    svuint32_t src_vec_u32 = svunpklo_u32(svunpklo_u16(src_vec_u8));
    svbool_t mask = svcmpne_u32(pg_32, src_vec_u32, ZERO_U32);
    svst1_f32(pg_32, dst + i, svsel_f32(mask, ONE_F32, ZERO_F32));
  }
}

template <>
inline Vectorized<float> fmadd(const Vectorized<float>& a, const Vectorized<float>& b, const Vectorized<float>& c) {
  return svmad_f32_x(ptrue, a, b, c);
}

template <>
Vectorized<float> inline fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return svmsb_f32_x(ptrue, a, b, c);
}

template <>
Vectorized<float> inline fmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return svnmsb_f32_x(ptrue, a, b, c);
}

template <>
Vectorized<float> inline fnmsub(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return svnmad_f32_x(ptrue, a, b, c);
}

#endif // defined(CPU_CAPABILITY_SVE)

} // namespace CPU_CAPABILITY
} // namespace at::vec
