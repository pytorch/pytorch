#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/irange.h>
#include <cmath>

#ifdef USE_FBGEMM
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wextra-semi")
#include <fbgemm/Fbgemm.h>
C10_DIAGNOSTIC_POP()
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
  Vectorized() {
    values = vmovq_n_f32(0);
  }
  Vectorized(svfloat32_t v) : values(svget_neonq(v)) {}
  Vectorized(float32x4_t v) : values(v) {}
  Vectorized(float val) {
    values = svget_neonq(svdup_n_f32(val));
  }
  Vectorized(float val0, float val1, float val2, float val3)
      : values{val0, val1, val2, val3} {}
  Vectorized(float (&arr)[4]) : Vectorized(arr[0], arr[1], arr[2], arr[3]) {}
  template <
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ float buffer[size()] = {vals...};
    values = vld1q_f32(buffer);
  }
  operator svfloat32_t() const {
    return svset_neonq(svundef_f32(), values);
  }
  operator float32x4_t() const {
    return values;
  }
  svfloat32_t valuesAsSve() const {
    return svset_neonq(svundef_f32(), values);
  }
  template <int64_t mask>
  static Vectorized<float> blend(
      const Vectorized<float>& a,
      const Vectorized<float>& b) {
    // Build an array of flags: each bit of element is 1 if the corresponding
    // bit in 'mask' is set, 0 otherwise.
    uint32x4_t maskArray = {
        (mask & 1ULL) ? 0xFFFFFFFF : 0,
        (mask & 2ULL) ? 0xFFFFFFFF : 0,
        (mask & 4ULL) ? 0xFFFFFFFF : 0,
        (mask & 8ULL) ? 0xFFFFFFFF : 0};
    // Use BSL to select elements from b where the mask is 1, else from a
    return vbslq_f32(maskArray, b.values, a.values);
  }
  static Vectorized<float> blendv(
      const Vectorized<float>& a,
      const Vectorized<float>& b,
      const Vectorized<float>& mask_) {
    return vbslq_f32(vreinterpretq_u32_f32(mask_.values), b.values, a.values);
  }
  template <typename step_t>
  static Vectorized<float> arange(
      float base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    const Vectorized<float> base_vec(base);
    const Vectorized<float> step_vec(step);
    const Vectorized<float> step_sizes(0.0f, 1.0f, 2.0f, 3.0f);
    return fmadd(step_sizes, step_vec, base_vec);
  }
  static Vectorized<float> set(
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
  // Implementation is improved from
  // https://github.com/ARM-software/ComputeLibrary/blob/v25.01/src/core/NEON/SVEMath.inl#L105

  inline float32x4_t vtaylor_polyq_for_log_f32(float32x4_t x) const {
    const float32x4_t log_tab_1 = vdupq_n_f32(-2.29561495781f);
    const float32x4_t log_tab_2 = vdupq_n_f32(-2.47071170807f);
    const float32x4_t log_tab_3 = vdupq_n_f32(-5.68692588806f);
    const float32x4_t log_tab_4 = vdupq_n_f32(-0.165253549814f);
    const float32x4_t log_tab_5 = vdupq_n_f32(5.17591238022f);
    const float32x4_t log_tab_6 = vdupq_n_f32(0.844007015228f);
    const float32x4_t log_tab_7 = vdupq_n_f32(4.58445882797f);
    const float32x4_t log_tab_8 = vdupq_n_f32(0.0141278216615f);

    float32x4_t A = vmlaq_f32(log_tab_1, log_tab_5, x);
    float32x4_t B = vmlaq_f32(log_tab_3, log_tab_7, x);
    float32x4_t C = vmlaq_f32(log_tab_2, log_tab_6, x);
    float32x4_t D = vmlaq_f32(log_tab_4, log_tab_8, x);
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x4 = vmulq_f32(x2, x2);
    float32x4_t res = vmlaq_f32(vmlaq_f32(A, B, x2), vmlaq_f32(C, D, x2), x4);
    return res;
  }

  inline float32x4_t vlogq_f32(float32x4_t x) const {
    const float32x4_t CONST_LN2 = vdupq_n_f32(0.6931471805f); // ln(2)

    // Extract exponent
    int32x4_t m = svget_neonq(svsub_n_s32_x(
        ptrue,
        svset_neonq(
            svundef_s32(),
            vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_f32(x), 23))),
        127));
    float32x4_t val = vreinterpretq_f32_s32(
        vsubq_s32(vreinterpretq_s32_f32(x), vshlq_n_s32(m, 23)));

    // Polynomial Approximation
    float32x4_t poly = vtaylor_polyq_for_log_f32(val);

    // Reconstruct
    poly = vmlaq_f32(poly, vcvtq_f32_s32(m), CONST_LN2);

    return poly;
  }

  inline float32x4_t vexpq_f32(float32x4_t x) const {
    const auto c1 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3f7ffff6)));
    const auto c2 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3efffedb)));
    const auto c3 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3e2aaf33)));
    const auto c4 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3d2b9f17)));
    const auto c5 = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(0x3c072010)));

    const auto shift = vreinterpretq_f32_u32(
        svget_neonq(svdup_n_u32(0x4b00007f))); // 2^23 + 127 = 0x1.0000fep23f
    const auto inv_ln2 = vreinterpretq_f32_u32(
        svget_neonq(svdup_n_u32(0x3fb8aa3b))); // 1 / ln(2) = 0x1.715476p+0f
    const auto neg_ln2_hi = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(
        0xbf317200))); // -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
    const auto neg_ln2_lo = vreinterpretq_f32_u32(svget_neonq(svdup_n_u32(
        0xb5bfbe8e))); // -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f

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
    const auto z = vfmaq_f32(shift, x, inv_ln2);
    const auto n = z - shift;
    const auto scale =
        vreinterpretq_f32_u32(vreinterpretq_u32_f32(z) << 23); // 2^n

    // The calculation of n * ln(2) is done using 2 steps to achieve accuracy
    // beyond FP32. This outperforms longer Taylor series (3-4 tabs) both in
    // term of accuracy and performance.
    const auto r_hi = vfmaq_f32(x, n, neg_ln2_hi);
    const auto r = vfmaq_f32(r_hi, n, neg_ln2_lo);

    // Compute the truncated Taylor series of e^r.
    //   poly = scale * (1 + c1 * r + c2 * r^2 + c3 * r^3 + c4 * r^4 + c5 * r^5)
    const auto r2 = r * r;

    const auto p1 = c1 * r;
    const auto p23 = vfmaq_f32(c2, c3, r);
    const auto p45 = vfmaq_f32(c4, c5, r);
    const auto p2345 = vfmaq_f32(p23, p45, r2);
    const auto p12345 = vfmaq_f32(p1, p2345, r2);

    auto poly = svset_neonq(svundef_f32(), vfmaq_f32(scale, p12345, scale));

    // Handle underflow and overflow.
    poly = svsel_f32(
        svcmplt_f32(svptrue_b8(), svset_neonq(svundef_f32(), x), min_input),
        zero,
        poly);
    poly = svsel_f32(
        svcmpgt_f32(svptrue_b8(), svset_neonq(svundef_f32(), x), max_input),
        inf,
        poly);

    return svget_neonq(poly);
  }

  static Vectorized<float> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return vld1q_f32(reinterpret_cast<const float*>(ptr));
    svbool_t pg = svwhilelt_b32(0ull, count);
    return svld1_f32(pg, reinterpret_cast<const float*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    if (count == size()) {
      vst1q_f32(reinterpret_cast<float*>(ptr), values);
    } else {
      svbool_t pg = svwhilelt_b32(0ull, count);
      svst1_f32(pg, reinterpret_cast<float*>(ptr), valuesAsSve());
    }
  }
  inline float operator[](int idx) const {
    return values[idx];
  }
  inline float operator[](int idx) {
    return values[idx];
  }
  int64_t zero_mask() const {
    uint32x4_t cmpReg = vceqzq_f32(values);
    uint16x4_t narrowedCmp = vmovn_u32(cmpReg);
    uint64x2_t extReg = svget_neonq(svbext_u64(
        svset_neonq(
            svundef_u64(),
            vreinterpretq_u64_u16(vcombine_u16(narrowedCmp, vdup_n_u16(0)))),
        svreinterpret_u64_u16(svdup_u16(1))));
    return extReg[0];
  }
  Vectorized<float> isnan() const {
    // NaN check
    return vreinterpretq_f32_u32(vmvnq_u32(vceqq_f32(values, values)));
  }
  bool has_inf_nan() const {
    return svptest_any(
        ptrue,
        svcmpuo_f32(
            ptrue,
            svset_neonq(svundef_f32(), vsubq_f32(values, values)),
            ZERO_F32));
  }
  Vectorized<float> map(float (*const f)(float)) const {
    float32x4_t result;
    result[0] = f(values[0]);
    result[1] = f(values[1]);
    result[2] = f(values[2]);
    result[3] = f(values[3]);
    return result;
  }
  Vectorized<float> map2(
      const Vectorized<float>& second,
      float (*const f)(float, float)) const {
    float32x4_t result;
    result[0] = f(values[0], second[0]);
    result[1] = f(values[1], second[1]);
    result[2] = f(values[2], second[2]);
    result[3] = f(values[3], second[3]);
    return result;
  }
  Vectorized<float> abs() const {
    return Vectorized<float>(vabsq_f32(values));
  }
  Vectorized<float> angle() const {
    auto zero = Vectorized<float>(0);
    auto pi = Vectorized<float>(c10::pi<float>);
    auto tmp = blendv(zero, pi, vcltzq_f32(*this));
    return blendv(tmp, *this, isnan());
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
  Vectorized<float> exp() const {
    return vexpq_f32(values);
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(exp2)
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(expm1)
  Vectorized<float> exp_u20() const {
    return exp();
  }
  Vectorized<float> fexp_u20() const {
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
  Vectorized<float> log() const {
    return vlogq_f32(values);
  }
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
    return vrndpq_f32(values);
  }
  Vectorized<float> floor() const {
    return vrndmq_f32(values);
  }
  Vectorized<float> neg() const {
    return Vectorized<float>(vnegq_f32(values));
  }
  Vectorized<float> round() const {
    return vrndiq_f32(values);
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(tan)
  // Implementation is inspired from
  // https://github.com/ARM-software/ComputeLibrary/blob/v25.01/src/core/NEON/SVEMath.inl#L179
  Vectorized<float> tanh() const {
    // Constants used for the tanh calculation.
    const float32x4_t CONST_2 = svget_neonq(svdup_n_f32(
        2.f)); // Constant 2.0f for the tanh formula (used in exp(2x)).
    const float32x4_t CONST_MIN_TANH = svget_neonq(svdup_n_f32(
        -10.f)); // Minimum threshold for input values to prevent overflow.
    const float32x4_t CONST_MAX_TANH = svget_neonq(svdup_n_f32(
        10.f)); // Maximum threshold for input values to prevent overflow.

    // Step 1: Clamp the values within the range [-10, 10] to prevent overflow
    // during exponentiation. The tanh function approaches ±1 rapidly as the
    // input grows large, so we limit the input range to avoid numerical
    // instability. vmaxq_f32 ensures values are greater than -10, and
    // vminq_f32 ensures they are less than 10.
    float32x4_t x =
        vminq_f32(vmaxq_f32(values, CONST_MIN_TANH), CONST_MAX_TANH);

    // Step 2: Calculate exp(2 * x), where x is the clamped value.
    // svmul_f32_z computes 2 * x, and vexpq_f32 computes the exponential of
    // the result.
    svfloat32_t exp2x =
        svset_neonq(svundef_f32(), vexpq_f32(vmulq_f32(CONST_2, x)));

    // Step 3: Calculate the numerator of the tanh function, which is exp(2x)
    // - 1.
    float32x4_t num = svget_neonq(svsub_n_f32_x(ptrue, exp2x, 1.f));

    // Step 4: Calculate the denominator of the tanh function, which is exp(2x)
    // + 1.
    float32x4_t den = svget_neonq(svadd_n_f32_x(ptrue, exp2x, 1.f));

    // Step 5: Calculate the tanh function as the ratio of the numerator and
    // denominator: num / den.
    float32x4_t tanh = vdivq_f32(num, den);

    // Return the calculated tanh values.
    return tanh;
  }
  Vectorized<float> trunc() const {
    return Vectorized<float>(vrndq_f32(values));
  }
  DEFINE_SLEEF_COMPATIBLE_UNARY_ELEMENTWISE_FUNC(lgamma)
  Vectorized<float> sqrt() const {
    return Vectorized<float>(vsqrtq_f32(values));
  }
  Vectorized<float> reciprocal() const {
    float32x4_t recip = vrecpeq_f32(values);
    recip = vmulq_f32(vrecpsq_f32(values, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(values, recip), recip);
    return recip;
  }
  Vectorized<float> rsqrt() const {
    float32x4_t sqrt_reciprocal = vrsqrteq_f32(values);
    sqrt_reciprocal = vmulq_f32(
        vrsqrtsq_f32(vmulq_f32(values, sqrt_reciprocal), sqrt_reciprocal),
        sqrt_reciprocal);
    sqrt_reciprocal = vmulq_f32(
        vrsqrtsq_f32(vmulq_f32(values, sqrt_reciprocal), sqrt_reciprocal),
        sqrt_reciprocal);

    return sqrt_reciprocal;
  }

  // pow(a, n) = exp(n * ln(a))
  Vectorized<float> pow(const Vectorized<float>& x) const {
    return vexpq_f32(vmulq_f32(vlogq_f32(values), x));
  }

  float reduce_add() const {
    return vaddvq_f32(values);
  }
  float reduce_max() const {
    return vmaxvq_f32(values);
  }

  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
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
Vectorized<float> inline Vectorized<float>::frac() const {
  return *this - this->trunc();
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
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

Vectorized<float> inline Vectorized<float>::eq(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this == other), 1));
}

Vectorized<float> inline Vectorized<float>::ne(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this != other), 1));
}

Vectorized<float> inline Vectorized<float>::gt(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this > other), 1));
}

Vectorized<float> inline Vectorized<float>::ge(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this >= other), 1));
}

Vectorized<float> inline Vectorized<float>::lt(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this < other), 1));
}

Vectorized<float> inline Vectorized<float>::le(
    const Vectorized<float>& other) const {
  return svreinterpret_f32_u32(
      svand_n_u32_x(ptrue, svreinterpret_u32_f32(*this <= other), 1));
}

template <>
inline void convert(const float* src, float* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<float>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_f32(src);
    auto vec2 = vld1q_f32(src + oneRegElemCount);
    auto vec3 = vld1q_f32(src + twoRegsElemCount);
    auto vec4 = vld1q_f32(src + (twoRegsElemCount + oneRegElemCount));
    vst1q_f32(dst, vec1);
    vst1q_f32(dst + oneRegElemCount, vec2);
    vst1q_f32(dst + twoRegsElemCount, vec3);
    vst1q_f32(dst + (twoRegsElemCount + oneRegElemCount), vec4);
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
inline void convert(const float* src, at::Half* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<float>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  __fp16* dstPtr = reinterpret_cast<__fp16*>(dst);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_f32(src);
    auto vec2 = vld1q_f32(src + oneRegElemCount);
    auto vec3 = vld1q_f32(src + twoRegsElemCount);
    auto vec4 = vld1q_f32(src + (twoRegsElemCount + oneRegElemCount));
    auto convertedVec1 = vcvt_high_f16_f32(vcvt_f16_f32(vec1), vec2);
    auto convertedVec2 = vcvt_high_f16_f32(vcvt_f16_f32(vec3), vec4);
    vst1q_f16(dstPtr, convertedVec1);
    vst1q_f16(dstPtr + twoRegsElemCount, convertedVec2);
    src += fourRegsElemCount;
    dstPtr += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0;
       --remainder) {
    *dstPtr = *src;
    src += 1;
    dstPtr += 1;
  }
}

template <>
inline void convert(const at::Half* src, float* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<float>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  constexpr uint64_t fourRegsElemCount = twoRegsElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  const __fp16* srcPtr = reinterpret_cast<const __fp16*>(src);
  for (uint64_t iters = count / fourRegsElemCount; iters > 0; --iters) {
    auto vec1 = vld1q_f16(srcPtr);
    auto vec2 = vld1q_f16(srcPtr + twoRegsElemCount);
    auto convertedVec1 = vcvt_f32_f16(vget_low_f16(vec1));
    auto convertedVec3 = vcvt_f32_f16(vget_low_f16(vec2));
    auto convertedVec2 = vcvt_high_f32_f16(vec1);
    auto convertedVec4 = vcvt_high_f32_f16(vec2);
    vst1q_f32(dst, convertedVec1);
    vst1q_f32(dst + oneRegElemCount, convertedVec2);
    vst1q_f32(dst + twoRegsElemCount, convertedVec3);
    vst1q_f32(dst + (twoRegsElemCount + oneRegElemCount), convertedVec4);
    srcPtr += fourRegsElemCount;
    dst += fourRegsElemCount;
  }
#pragma clang loop vectorize(disable)
#pragma clang loop unroll(disable)
  for (uint64_t remainder = count % fourRegsElemCount; remainder > 0;
       --remainder) {
    *dst = *srcPtr;
    srcPtr += 1;
    dst += 1;
  }
}

template <>
inline void convert(const bool* src, float* dst, int64_t n) {
  constexpr uint64_t oneRegElemCount = Vectorized<float>::size();
  constexpr uint64_t twoRegsElemCount = oneRegElemCount * 2;
  const uint64_t count = static_cast<uint64_t>(n);
  const uint64_t remainder = count % twoRegsElemCount;
  const uint8_t* srcPtr = reinterpret_cast<const uint8_t*>(src);
  for (uint64_t iters = count / twoRegsElemCount; iters > 0; --iters) {
    auto vec1 = svget_neonq(svld1ub_u32(svptrue_b8(), srcPtr));
    auto vec2 =
        svget_neonq(svld1ub_u32(svptrue_b8(), srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_u32(vec1, vec1);
    auto vec2Mask = vtstq_u32(vec2, vec2);
    vec1Mask = svget_neonq(svand_n_u32_x(
        svptrue_b8(), svset_neonq(svundef_u32(), vec1Mask), 0x3f800000));
    vec2Mask = svget_neonq(svand_n_u32_x(
        svptrue_b8(), svset_neonq(svundef_u32(), vec2Mask), 0x3f800000));
    vst1q_f32(dst, vreinterpretq_f32_u32(vec1Mask));
    vst1q_f32(dst + oneRegElemCount, vreinterpretq_f32_u32(vec2Mask));
    srcPtr += twoRegsElemCount;
    dst += twoRegsElemCount;
  }
  if (remainder > 0) {
    svbool_t pa = svwhilelt_b32_u64(0, remainder);
    svbool_t pb = svwhilelt_b32_u64(oneRegElemCount, remainder);
    auto vec1 = svget_neonq(svld1ub_u32(pa, srcPtr));
    auto vec2 = svget_neonq(svld1ub_u32(pb, srcPtr + oneRegElemCount));
    auto vec1Mask = vtstq_u32(vec1, vec1);
    auto vec2Mask = vtstq_u32(vec2, vec2);
    vec1Mask = svget_neonq(
        svand_n_u32_x(pa, svset_neonq(svundef_u32(), vec1Mask), 0x3f800000));
    vec2Mask = svget_neonq(
        svand_n_u32_x(pb, svset_neonq(svundef_u32(), vec2Mask), 0x3f800000));
    svst1_f32(
        pa, dst, svset_neonq(svundef_f32(), vreinterpretq_f32_u32(vec1Mask)));
    svst1_f32(
        pb,
        dst + oneRegElemCount,
        svset_neonq(svundef_f32(), vreinterpretq_f32_u32(vec2Mask)));
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
Vectorized<float> inline fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c) {
  return Vectorized<float>(vfmsq_f32(c, a, b));
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
  auto tmp4 = Vectorized<float>(vexpq_f32(neg_pow_2));
  auto tmp5 = tmp4 ^ neg_zero_vec;
  // erf(x) = sign(x) * (1 - r * t * exp(- x * x))
  auto tmp6 = t * tmp5;
  auto tmp7 = fmadd(tmp6, r, one_vec);
  return tmp7 ^ sign_mask;
}

#ifdef USE_FBGEMM

template <>
inline void transpose_mxn<float>(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst,
    int M,
    int N) {
  fbgemm::transpose_simd<float>(M, N, src, ld_src, dst, ld_dst);
}

template <
    typename T,
    int M,
    int N,
    typename std::enable_if_t<std::is_same_v<T, float>, int> = 0>
inline void transpose_mxn(
    const float* src,
    int64_t ld_src,
    float* dst,
    int64_t ld_dst) {
  transpose_mxn<float>(src, ld_src, dst, ld_dst, M, N);
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
