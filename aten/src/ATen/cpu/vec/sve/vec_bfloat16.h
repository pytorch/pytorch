#pragma once

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/sve/sve_helper.h>
#include <ATen/cpu/vec/sve/vec_float.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/bit_cast.h>
#include <cmath>
namespace at {
namespace vec {
// Note [CPU_CAPABILITY namespace]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This header, and all of its subheaders, will be compiled with
// different architecture flags for each supported set of vector
// intrinsics. So we need to make sure they aren't inadvertently
// linked together. We do this by declaring objects in an `inline
// namespace` which changes the name mangling, but can still be
// accessed as `at::vec`.
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_SVE256) && defined(__ARM_FEATURE_BF16)

template <>
struct is_vec_specialized_for<BFloat16> : std::bool_constant<true> {};

template <>
class Vectorized<BFloat16> {
 private:
  vls_bfloat16_t values;

 public:
  using value_type = BFloat16;
  using size_type = int;

  static constexpr size_type size() {
    return VECTOR_WIDTH / sizeof(BFloat16);
  }

  Vectorized();
  Vectorized(svbfloat16_t v) : values(v) {}
  Vectorized(int val);
  Vectorized(BFloat16 val);

  template <
      typename... Args,
      typename = std::enable_if_t<(sizeof...(Args) == size())>>
  Vectorized(Args... vals) {
    __at_align__ BFloat16 buffer[size()] = {vals...};
    values = svld1_bf16(ptrue, reinterpret_cast<const bfloat16_t*>(buffer));
  }

  operator svbfloat16_t() const {
    return values;
  }
  static Vectorized<BFloat16> blendv(
      const Vectorized<BFloat16>& a,
      const Vectorized<BFloat16>& b,
      const Vectorized<BFloat16>& mask_) {
    svbool_t mask =
        svcmpeq_s16(ptrue, svreinterpret_s16_bf16(mask_), ALL_S16_TRUE_MASK);
    return svsel_bf16(mask, b, a);
  }
  template <typename step_t>
  static Vectorized<BFloat16> arange(
      BFloat16 base = 0.f,
      step_t step = static_cast<step_t>(1)) {
    __at_align__ BFloat16 buffer[size()];
    for (int64_t i = 0; i < size(); i++) {
      buffer[i] = base + i * step;
    }
    return svld1_bf16(ptrue, reinterpret_cast<bfloat16_t*>(buffer));
  }
  static Vectorized<BFloat16> set(
      const Vectorized<BFloat16>& a,
      const Vectorized<BFloat16>& b,
      int64_t count = size()) {
    if (count == 0) {
      return a;
    } else if (count < size()) {
      return svsel_bf16(svwhilelt_b16(0ull, count), b, a);
    }
    return b;
  }
  static Vectorized<BFloat16> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return svld1_bf16(ptrue, reinterpret_cast<const bfloat16_t*>(ptr));
    svbool_t pg = svwhilelt_b16(0ull, count);
    return svld1_bf16(pg, reinterpret_cast<const bfloat16_t*>(ptr));
  }
  void store(void* ptr, int64_t count = size()) const {
    __at_align__ bfloat16_t tmp[size()];
    std::memset(tmp, 0, sizeof(tmp));
    if (count == size()) {
      svst1_bf16(ptrue, reinterpret_cast<bfloat16_t*>(tmp), values);
    } else {
      svbool_t pg = svwhilelt_b16(0ull, count);
      svst1_bf16(pg, reinterpret_cast<bfloat16_t*>(tmp), values);
    }
    std::memcpy(
        reinterpret_cast<bfloat16_t*>(ptr),
        reinterpret_cast<const bfloat16_t*>(tmp),
        count * sizeof(bfloat16_t));
  }
  const BFloat16& operator[](int idx) const = delete;
  BFloat16& operator[](int idx) = delete;
  int64_t zero_mask() const {
    int64_t mask = 0;
    // returns an integer mask where all zero elements are translated to
    // 1-bit and others are translated to 0-bit int64_t mask = 0;
    __at_align__ int16_t mask_array[size()];

    svbool_t svbool_mask =
        svcmpeq_f16(ptrue, svreinterpret_f16_bf16(values), ZERO_F16);
    svst1_s16(
        ptrue,
        mask_array,
        svsel_s16(svbool_mask, ALL_S16_TRUE_MASK, ALL_S16_FALSE_MASK));
    for (int64_t i = 0; i < size(); ++i) {
      if (mask_array[i])
        mask |= (1ull << i);
    }
    return mask;
  }
  Vectorized<BFloat16> isnan() const;
  bool has_inf_nan() const;
  Vectorized<BFloat16> map(BFloat16 (*f)(BFloat16)) const {
    __at_align__ BFloat16 tmp[size()];
    store(tmp);
    for (int64_t i = 0; i < size(); ++i) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vectorized<BFloat16> abs() const {
    auto mask = svdup_n_u16(0x7FFF);
    auto vals = svreinterpret_u16_bf16(values);
    vals = svand_u16_x(ptrue, vals, mask);
    return svreinterpret_bf16_u16(vals);
  }
  Vectorized<BFloat16> angle() const;
  Vectorized<BFloat16> real() const {
    return values;
  }
  Vectorized<BFloat16> imag() const {
    return Vectorized<BFloat16>(0.f);
  }
  Vectorized<BFloat16> conj() const {
    return values;
  }
  Vectorized<BFloat16> acos() const;
  Vectorized<BFloat16> acosh() const;
  Vectorized<BFloat16> asin() const;
  Vectorized<BFloat16> atan() const;
  Vectorized<BFloat16> atanh() const;
  Vectorized<BFloat16> atan2(const Vectorized<BFloat16>& b) const;
  Vectorized<BFloat16> copysign(const Vectorized<BFloat16>& sign) const;
  Vectorized<BFloat16> erf() const;
  Vectorized<BFloat16> erfc() const;
  Vectorized<BFloat16> erfinv() const;
  Vectorized<BFloat16> exp() const;
  Vectorized<BFloat16> exp2() const;
  Vectorized<BFloat16> expm1() const;
  Vectorized<BFloat16> exp_u20() const {
    return exp();
  }
  Vectorized<BFloat16> fexp_u20() const {
    return exp();
  }
  Vectorized<BFloat16> fmod(const Vectorized<BFloat16>& q) const;
  Vectorized<BFloat16> hypot(const Vectorized<BFloat16>& b) const;
  Vectorized<BFloat16> i0() const;
  Vectorized<BFloat16> i0e() const;
  Vectorized<BFloat16> digamma() const;
  Vectorized<BFloat16> igamma(const Vectorized<BFloat16>& x) const;
  Vectorized<BFloat16> igammac(const Vectorized<BFloat16>& x) const;
  Vectorized<BFloat16> nextafter(const Vectorized<BFloat16>& b) const;
  Vectorized<BFloat16> log() const;
  Vectorized<BFloat16> log2() const;
  Vectorized<BFloat16> log10() const;
  Vectorized<BFloat16> log1p() const;
  Vectorized<BFloat16> frac() const;
  Vectorized<BFloat16> sin() const;
  Vectorized<BFloat16> sinh() const;
  Vectorized<BFloat16> cos() const;
  Vectorized<BFloat16> cosh() const;
  Vectorized<BFloat16> ceil() const;
  Vectorized<BFloat16> floor() const;
  Vectorized<BFloat16> neg() const {
    auto mask = svdup_n_u16(0x8000);
    auto vals = svreinterpret_u16_bf16(values);
    vals = sveor_u16_x(ptrue, vals, mask);
    return svreinterpret_bf16_u16(vals);
  };
  Vectorized<BFloat16> round() const;
  Vectorized<BFloat16> tan() const;
  Vectorized<BFloat16> tanh() const;
  Vectorized<BFloat16> trunc() const;
  Vectorized<BFloat16> lgamma() const;
  Vectorized<BFloat16> sqrt() const;
  Vectorized<BFloat16> reciprocal() const;
  Vectorized<BFloat16> rsqrt() const;
  Vectorized<BFloat16> pow(const Vectorized<BFloat16>& b) const;
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<BFloat16> operator==(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> operator!=(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> operator<(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> operator<=(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> operator>(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> operator>=(const Vectorized<BFloat16>& other) const;

  Vectorized<BFloat16> eq(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ne(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> gt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> ge(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> lt(const Vectorized<BFloat16>& other) const;
  Vectorized<BFloat16> le(const Vectorized<BFloat16>& other) const;
};

#if defined(__GNUC__) && __GNUC__ == 14
// Workaround for gcc-14.2.0 ICE during RTL pass: vregs when compiling for SVE
__attribute__((optimize("no-tree-vectorize")))
#endif
inline std::tuple<Vectorized<float>, Vectorized<float>>
convert_bfloat16_float(const Vectorized<c10::BFloat16>& a) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  auto zero = svreinterpret_bf16_f32(svdup_n_f32(0.0f));
  auto bf16_vec1 = svzip1_bf16(zero, a);
  auto bf16_vec2 = svzip2_bf16(zero, a);
  auto x1 = svreinterpret_f32_bf16(bf16_vec1);
  auto x2 = svreinterpret_f32_bf16(bf16_vec2);
  return {Vectorized<float>(x1), Vectorized<float>(x2)};
}

inline Vectorized<c10::BFloat16> convert_float_bfloat16(
    const Vectorized<float>& a,
    const Vectorized<float>& b) {
  static_assert(
      Vectorized<c10::BFloat16>::size() == 2 * Vectorized<float>::size());
  svbfloat16_t x1 = svcvt_bf16_f32_z(ptrue, a);
  svbfloat16_t x2 = svcvt_bf16_f32_z(ptrue, b);
  return Vectorized<c10::BFloat16>(svuzp1_bf16(x1, x2));
}

inline void load_fp32_from_bf16(const BFloat16* data, Vectorized<float>& out) {
  __at_align__ float values[Vectorized<float>::size()];
  for (const auto k : c10::irange(Vectorized<float>::size())) {
    values[k] = data[k];
  }
  out = Vectorized<float>::loadu(values);
}

inline void load_fp32_from_bf16(
    const BFloat16* data,
    Vectorized<float>& out1,
    Vectorized<float>& out2) {
  Vectorized<BFloat16> bf16_vec = Vectorized<BFloat16>::loadu(data);
  auto floats = convert_bfloat16_float(bf16_vec);
  out1 = std::get<0>(floats);
  out2 = std::get<1>(floats);
}

template <typename Op>
Vectorized<c10::BFloat16> binary_operator_via_float(
    Op op,
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  const auto [a_float_low, a_float_high] = convert_bfloat16_float(a);
  const auto [b_float_low, b_float_high] = convert_bfloat16_float(b);
  return convert_float_bfloat16(
      op(a_float_low, b_float_low), op(a_float_high, b_float_high));
}

template <>
Vectorized<c10::BFloat16> inline operator+(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::plus<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator-(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::minus<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator*(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::multiplies<Vectorized<float>>(), a, b);
}

template <>
Vectorized<c10::BFloat16> inline operator/(
    const Vectorized<c10::BFloat16>& a,
    const Vectorized<c10::BFloat16>& b) {
  return binary_operator_via_float(std::divides<Vectorized<float>>(), a, b);
}

inline Vectorized<BFloat16>::Vectorized() {
  const short zero = 0;
  values = svdup_n_bf16(c10::bit_cast<bfloat16_t>(zero));
}

inline Vectorized<BFloat16>::Vectorized(int val) {
  auto vals_f = svdup_n_f32(val);
  values = convert_float_bfloat16(vals_f, vals_f);
}

inline Vectorized<BFloat16>::Vectorized(BFloat16 val) {
  auto vals_f = svdup_n_f32((float)val);
  values = convert_float_bfloat16(vals_f, vals_f);
}

bool inline Vectorized<c10::BFloat16>::has_inf_nan() const {
  auto [v1, v2] = convert_bfloat16_float(values);
  return v1.has_inf_nan() || v2.has_inf_nan();
}
// frac. Implement this here so we can use subtraction
Vectorized<BFloat16> inline Vectorized<BFloat16>::frac() const {
  return *this - this->trunc();
}

#define DEFINE_BF16_FUNC_VIA_FLOAT(func_name)                           \
  Vectorized<BFloat16> inline Vectorized<BFloat16>::func_name() const { \
    auto [v1, v2] = convert_bfloat16_float(*this);                      \
    v1 = v1.func_name();                                                \
    v2 = v2.func_name();                                                \
    return convert_float_bfloat16(v1, v2);                              \
  }

#define DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(func_name)            \
  Vectorized<BFloat16> inline Vectorized<BFloat16>::func_name( \
      const Vectorized<BFloat16>& a) const {                   \
    auto [v1, v2] = convert_bfloat16_float(*this);             \
    auto [v3, v4] = convert_bfloat16_float(a);                 \
    v1 = v1.func_name(v3);                                     \
    v2 = v2.func_name(v4);                                     \
    return convert_float_bfloat16(v1, v2);                     \
  }

DEFINE_BF16_FUNC_VIA_FLOAT(isnan);
DEFINE_BF16_FUNC_VIA_FLOAT(angle);
DEFINE_BF16_FUNC_VIA_FLOAT(acos);
DEFINE_BF16_FUNC_VIA_FLOAT(acosh);
DEFINE_BF16_FUNC_VIA_FLOAT(asin);
DEFINE_BF16_FUNC_VIA_FLOAT(atan);
DEFINE_BF16_FUNC_VIA_FLOAT(atanh);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(atan2);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(copysign);
DEFINE_BF16_FUNC_VIA_FLOAT(erf);
DEFINE_BF16_FUNC_VIA_FLOAT(erfc);
DEFINE_BF16_FUNC_VIA_FLOAT(exp);
DEFINE_BF16_FUNC_VIA_FLOAT(exp2);
DEFINE_BF16_FUNC_VIA_FLOAT(expm1);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(fmod);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(hypot);
DEFINE_BF16_FUNC_VIA_FLOAT(i0);
DEFINE_BF16_FUNC_VIA_FLOAT(i0e);
DEFINE_BF16_FUNC_VIA_FLOAT(digamma);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(igamma);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(igammac);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(nextafter);
DEFINE_BF16_FUNC_VIA_FLOAT(log);
DEFINE_BF16_FUNC_VIA_FLOAT(log2);
DEFINE_BF16_FUNC_VIA_FLOAT(log10);
DEFINE_BF16_FUNC_VIA_FLOAT(log1p);
DEFINE_BF16_FUNC_VIA_FLOAT(sin);
DEFINE_BF16_FUNC_VIA_FLOAT(sinh);
DEFINE_BF16_FUNC_VIA_FLOAT(cos);
DEFINE_BF16_FUNC_VIA_FLOAT(cosh);
DEFINE_BF16_FUNC_VIA_FLOAT(ceil);
DEFINE_BF16_FUNC_VIA_FLOAT(floor);
DEFINE_BF16_FUNC_VIA_FLOAT(round);
DEFINE_BF16_FUNC_VIA_FLOAT(tan);
DEFINE_BF16_FUNC_VIA_FLOAT(tanh);
DEFINE_BF16_FUNC_VIA_FLOAT(trunc);
DEFINE_BF16_FUNC_VIA_FLOAT(lgamma);
DEFINE_BF16_FUNC_VIA_FLOAT(sqrt);
DEFINE_BF16_FUNC_VIA_FLOAT(reciprocal);
DEFINE_BF16_FUNC_VIA_FLOAT(rsqrt);
DEFINE_BF16_FUNC_VIA_FLOAT_W_ARG(pow);

Vectorized<BFloat16> inline Vectorized<BFloat16>::operator==(
    const Vectorized<BFloat16>& other) const {
  auto [f1, f2] = convert_bfloat16_float(values);
  auto [f3, f4] = convert_bfloat16_float(other);
  svbool_t mask1 = svcmpeq_f32(ptrue, f1, f3);
  svbool_t mask2 = svcmpeq_f32(ptrue, f2, f4);
  auto res1 = svsel_f32(mask1, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  auto res2 = svsel_f32(mask2, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);

  auto bf16_1 = svreinterpret_bf16_f32(res1);
  auto bf16_2 = svreinterpret_bf16_f32(res2);
  return svuzp1_bf16(bf16_1, bf16_2);
}
Vectorized<BFloat16> inline Vectorized<BFloat16>::operator!=(
    const Vectorized<BFloat16>& other) const {
  auto [f1, f2] = convert_bfloat16_float(values);
  auto [f3, f4] = convert_bfloat16_float(other);
  svbool_t mask1 = svcmpne_f32(ptrue, f1, f3);
  svbool_t mask2 = svcmpne_f32(ptrue, f2, f4);
  auto res1 = svsel_f32(mask1, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);
  auto res2 = svsel_f32(mask2, ALL_F32_TRUE_MASK, ALL_F32_FALSE_MASK);

  auto bf16_1 = svreinterpret_bf16_f32(res1);
  auto bf16_2 = svreinterpret_bf16_f32(res2);
  return svuzp1_bf16(bf16_1, bf16_2);
}
Vectorized<BFloat16> inline Vectorized<BFloat16>::operator>(
    const Vectorized<BFloat16>& other) const {
  auto [v1, v2] = convert_bfloat16_float(*this);
  auto [v3, v4] = convert_bfloat16_float(other);
  return convert_float_bfloat16(v1 > v3, v2 > v4);
}
Vectorized<BFloat16> inline Vectorized<BFloat16>::operator>=(
    const Vectorized<BFloat16>& other) const {
  auto [v1, v2] = convert_bfloat16_float(*this);
  auto [v3, v4] = convert_bfloat16_float(other);
  return convert_float_bfloat16(v1 >= v3, v2 >= v4);
}
Vectorized<BFloat16> inline Vectorized<BFloat16>::operator<(
    const Vectorized<BFloat16>& other) const {
  auto [v1, v2] = convert_bfloat16_float(*this);
  auto [v3, v4] = convert_bfloat16_float(other);
  return convert_float_bfloat16(v1 < v3, v2 < v4);
}
Vectorized<BFloat16> inline Vectorized<BFloat16>::operator<=(
    const Vectorized<BFloat16>& other) const {
  auto [v1, v2] = convert_bfloat16_float(*this);
  auto [v3, v4] = convert_bfloat16_float(other);
  return convert_float_bfloat16(v1 <= v3, v2 <= v4);
}

// Implements the IEEE 754 201X `maximum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline maximum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&maximum),
      a,
      b);
}

// Implements the IEEE 754 201X `minimum` operation, which propagates NaN if
// either input is a NaN.
template <>
Vectorized<BFloat16> inline minimum(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&minimum),
      a,
      b);
}

template <>
Vectorized<BFloat16> inline clamp_max(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& max) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&clamp_max),
      a,
      max);
}

template <>
Vectorized<BFloat16> inline clamp_min(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min) {
  return binary_operator_via_float(
      static_cast<Vectorized<float> (*)(
          const Vectorized<float>&, const Vectorized<float>&)>(&clamp_min),
      a,
      min);
}

template <>
Vectorized<BFloat16> inline clamp(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& min,
    const Vectorized<BFloat16>& max) {
  return clamp_min(clamp_max(a, max), min);
}

template <>
Vectorized<BFloat16> inline operator&(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return svreinterpret_bf16_u16(
      svand_u16_x(ptrue, svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
}

template <>
Vectorized<BFloat16> inline operator|(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return svreinterpret_bf16_u16(
      svorr_u16_x(ptrue, svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
}

template <>
Vectorized<BFloat16> inline operator^(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b) {
  return svreinterpret_bf16_u16(
      sveor_u16_x(ptrue, svreinterpret_u16_bf16(a), svreinterpret_u16_bf16(b)));
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::eq(
    const Vectorized<BFloat16>& other) const {
  return (*this == other) & Vectorized<BFloat16>(1.0f);
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::ne(
    const Vectorized<BFloat16>& other) const {
  return (*this != other) & Vectorized<BFloat16>(1.0f);
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::gt(
    const Vectorized<BFloat16>& other) const {
  return (*this > other) & Vectorized<BFloat16>(1.0f);
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::ge(
    const Vectorized<BFloat16>& other) const {
  return (*this >= other) & Vectorized<BFloat16>(1.0f);
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::lt(
    const Vectorized<BFloat16>& other) const {
  return (*this < other) & Vectorized<BFloat16>(1.0f);
}

Vectorized<BFloat16> inline Vectorized<BFloat16>::le(
    const Vectorized<BFloat16>& other) const {
  return (*this <= other) & Vectorized<BFloat16>(1.0f);
}

template <>
inline void convert(const BFloat16* src, BFloat16* dst, int64_t n) {
  const int64_t fraction = n % Vectorized<BFloat16>::size();
#pragma unroll
  for (int64_t i = 0; i < n - fraction; i += Vectorized<BFloat16>::size()) {
    svst1_bf16(
        ptrue,
        const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(dst)) + i,
        svldnt1_bf16(
            ptrue,
            const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(src)) +
                i));
  }
#pragma unroll
  for (int64_t i = n - fraction; i < n; i += Vectorized<BFloat16>::size()) {
    svbool_t pg = svwhilelt_b16(i, n);
    svst1_bf16(
        pg,
        const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(dst)) + i,
        svldnt1_bf16(
            pg,
            const_cast<bfloat16_t*>(reinterpret_cast<const bfloat16_t*>(src)) +
                i));
  }
}

template <>
Vectorized<BFloat16> inline fmadd(
    const Vectorized<BFloat16>& a,
    const Vectorized<BFloat16>& b,
    const Vectorized<BFloat16>& c) {
  return a * b + c;
}

#endif // defined(CPU_CAPABILITY_SVE) && defined(__ARM_FEATURE_BF16)

} // namespace CPU_CAPABILITY
} // namespace vec
} // namespace at
