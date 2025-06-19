#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#include <c10/util/complex.h>
#include <c10/util/irange.h>
#if defined(CPU_CAPABILITY_AVX2)
#define SLEEF_STATIC_LIBS
#include <sleef.h>
#endif

namespace at::vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX2)

template <>
struct is_vec_specialized_for<c10::complex<float>> : std::bool_constant<true> {
};

template <>
class Vectorized<c10::complex<float>> {
 private:
  __m256 values;

 public:
  using value_type = c10::complex<float>;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  Vectorized(__m256 v) : values(v) {}
  Vectorized(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    values = _mm256_setr_ps(
        real_value,
        imag_value,
        real_value,
        imag_value,
        real_value,
        imag_value,
        real_value,
        imag_value);
  }
  Vectorized(
      c10::complex<float> val1,
      c10::complex<float> val2,
      c10::complex<float> val3,
      c10::complex<float> val4) {
    values = _mm256_setr_ps(
        val1.real(),
        val1.imag(),
        val2.real(),
        val2.imag(),
        val3.real(),
        val3.imag(),
        val4.real(),
        val4.imag());
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<c10::complex<float>> blend(
      const Vectorized<c10::complex<float>>& a,
      const Vectorized<c10::complex<float>>& b) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    static_assert(mask > -1 && mask < 16, "Unexpected mask range");
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm256_blend_ps(
            a.values, b.values, 0x03); // b0000 0001 = b0000 0011
      case 2:
        return _mm256_blend_ps(
            a.values, b.values, 0x0C); // b0000 0010 = b0000 1100
      case 3:
        return _mm256_blend_ps(
            a.values, b.values, 0x0F); // b0000 0011 = b0000 1111
      case 4:
        return _mm256_blend_ps(
            a.values, b.values, 0x30); // b0000 0100 = b0011 0000
      case 5:
        return _mm256_blend_ps(
            a.values, b.values, 0x33); // b0000 0101 = b0011 0011
      case 6:
        return _mm256_blend_ps(
            a.values, b.values, 0x3C); // b0000 0110 = b0011 1100
      case 7:
        return _mm256_blend_ps(
            a.values, b.values, 0x3F); // b0000 0111 = b0011 1111
      case 8:
        return _mm256_blend_ps(
            a.values, b.values, 0xC0); // b0000 1000 = b1100 0000
      case 9:
        return _mm256_blend_ps(
            a.values, b.values, 0xC3); // b0000 1001 = b1100 0011
      case 10:
        return _mm256_blend_ps(
            a.values, b.values, 0xCC); // b0000 1010 = b1100 1100
      case 11:
        return _mm256_blend_ps(
            a.values, b.values, 0xCF); // b0000 1011 = b1100 1111
      case 12:
        return _mm256_blend_ps(
            a.values, b.values, 0xF0); // b0000 1100 = b1111 0000
      case 13:
        return _mm256_blend_ps(
            a.values, b.values, 0xF3); // b0000 1101 = b1111 0011
      case 14:
        return _mm256_blend_ps(
            a.values, b.values, 0xFC); // b0000 1110 = b1111 1100
      default:
        break;
    }
    return b;
  }
  static Vectorized<c10::complex<float>> blendv(
      const Vectorized<c10::complex<float>>& a,
      const Vectorized<c10::complex<float>>& b,
      const Vectorized<c10::complex<float>>& mask) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm256_unpacklo_ps(mask.values, mask.values);
    return _mm256_blendv_ps(a.values, b.values, mask_);
  }
  template <typename step_t>
  static Vectorized<c10::complex<float>> arange(
      c10::complex<float> base = 0.,
      step_t step = static_cast<step_t>(1)) {
    return Vectorized<c10::complex<float>>(
        base,
        base + step,
        base + c10::complex<float>(2) * step,
        base + c10::complex<float>(3) * step);
  }
  static Vectorized<c10::complex<float>> set(
      const Vectorized<c10::complex<float>>& a,
      const Vectorized<c10::complex<float>>& b,
      int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
      case 2:
        return blend<3>(a, b);
      case 3:
        return blend<7>(a, b);
    }
    return b;
  }
  static Vectorized<c10::complex<float>> loadu(
      const void* ptr,
      int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));

    __at_align__ float tmp_values[2 * size()];
    // Ensure uninitialized memory does not change the output value See
    // https://github.com/pytorch/pytorch/issues/32502 for more details. We do
    // not initialize arrays to zero using "={0}" because gcc would compile it
    // to two instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(2 * size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const float*>(ptr),
        count * sizeof(c10::complex<float>));
    return _mm256_load_ps(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_ps(reinterpret_cast<float*>(ptr), values);
    } else if (count > 0) {
      float tmp_values[2 * size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
    }
  }
  const c10::complex<float>& operator[](int idx) const = delete;
  c10::complex<float>& operator[](int idx) = delete;
  Vectorized<c10::complex<float>> map(
      c10::complex<float> (*const f)(const c10::complex<float>&)) const {
    __at_align__ c10::complex<float> tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  __m256 abs_2_() const {
    auto val_2 = _mm256_mul_ps(values, values); // a*a     b*b
    auto ret = _mm256_hadd_ps(val_2, val_2); // a*a+b*b a*a+b*b
    return _mm256_permute_ps(ret, 0xD8);
  }
  __m256 abs_() const {
    auto real = _mm256_moveldup_ps(values); // real real
    auto imag = _mm256_movehdup_ps(values); // imag imag
    return Sleef_hypotf8_u05(real, imag); // abs  abs
  }
  Vectorized<c10::complex<float>> abs() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000));
    return _mm256_and_ps(abs_(), real_mask); // abs     0
  }
  __m256 angle_() const {
    // angle = atan2(b/a)
    auto b_a = _mm256_permute_ps(values, 0xB1); // b        a
    return Sleef_atan2f8_u10(values, b_a); // 90-angle angle
  }
  Vectorized<c10::complex<float>> angle() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000));
    auto angle = _mm256_permute_ps(angle_(), 0xB1); // angle    90-angle
    return _mm256_and_ps(angle, real_mask); // angle    0
  }
  Vectorized<c10::complex<float>> sgn() const {
    auto abs = abs_();
    auto zero = _mm256_setzero_ps();
    auto mask = _mm256_cmp_ps(abs, zero, _CMP_EQ_OQ);
    auto div = _mm256_div_ps(values, abs);
    return _mm256_blendv_ps(div, zero, mask);
  }
  __m256 real_() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000));
    return _mm256_and_ps(values, real_mask);
  }
  Vectorized<c10::complex<float>> real() const {
    return real_();
  }
  __m256 imag_() const {
    const __m256 imag_mask = _mm256_castsi256_ps(_mm256_setr_epi32(
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF,
        0x00000000,
        0xFFFFFFFF));
    return _mm256_and_ps(values, imag_mask);
  }
  Vectorized<c10::complex<float>> imag() const {
    return _mm256_permute_ps(imag_(), 0xB1); // b        a
  }
  __m256 conj_() const {
    const __m256 sign_mask =
        _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_ps(values, sign_mask); // a       -b
  }
  Vectorized<c10::complex<float>> conj() const {
    return conj_();
  }
  Vectorized<c10::complex<float>> log() const {
    // Most trigonomic ops use the log() op to improve complex number
    // performance.
    return map(std::log);
  }
  Vectorized<c10::complex<float>> log2() const {
    const __m256 log2_ = _mm256_set1_ps(std::log(2));
    return _mm256_div_ps(log(), log2_);
  }
  Vectorized<c10::complex<float>> log10() const {
    const __m256 log10_ = _mm256_set1_ps(std::log(10));
    return _mm256_div_ps(log(), log10_);
  }
  Vectorized<c10::complex<float>> log1p() const {
    return map(std::log1p);
  }
  Vectorized<c10::complex<float>> asin() const {
    // TODO: The vectorized implementation requires special handling for the
    // case where real number/imag number is 0/Inf/NaN.
    // // asin(x)
    // // = -i*ln(iz + sqrt(1 -z^2))
    // // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    // const __m256 one = _mm256_set1_ps(1);

    // auto conj = conj_();
    // auto b_a = _mm256_permute_ps(conj, 0xB1);                         //-b a
    // auto ab = _mm256_mul_ps(conj, b_a);                               //-ab
    // -ab auto im = _mm256_add_ps(ab, ab); //-2ab      -2ab

    // auto val_2 = _mm256_mul_ps(values, values);                       // a*a
    // b*b auto re = _mm256_hsub_ps(val_2, _mm256_permute_ps(val_2, 0xB1));  //
    // a*a-b*b  b*b-a*a re = _mm256_permute_ps(re, 0xD8); re =
    // _mm256_sub_ps(one, re);

    // auto root = Vectorized(_mm256_blend_ps(re, im, 0xAA)).sqrt(); //sqrt(re +
    // i*im) auto ln = Vectorized(_mm256_add_ps(b_a, root)).log(); //ln(iz +
    // sqrt()) return Vectorized(_mm256_permute_ps(ln.values, 0xB1)).conj();
    // //-i*ln()
    return map(std::asin);
  }
  Vectorized<c10::complex<float>> acos() const {
    return map(std::acos);
  }
  Vectorized<c10::complex<float>> atan() const;
  Vectorized<c10::complex<float>> atanh() const {
    return map(std::atanh);
  }
  Vectorized<c10::complex<float>> exp() const {
    // TODO: The vectorized implementation requires special handling for the
    // case where real number/imag number is 0/Inf/NaN.
    // //exp(a + bi)
    // // = exp(a)*(cos(b) + sin(b)i)
    // auto exp = Sleef_expf8_u10(values); //exp(a)           exp(b) exp =
    // _mm256_blend_ps(exp, _mm256_permute_ps(exp, 0xB1), 0xAA);   //exp(a)
    // exp(a)

    // auto sin_cos = Sleef_sincosf8_u10(values); //[sin(a), cos(a)] [sin(b),
    // cos(b)] auto cos_sin = _mm256_blend_ps(_mm256_permute_ps(sin_cos.y,
    // 0xB1),
    //                                sin_cos.x, 0xAA); //cos(b) sin(b)
    // return _mm256_mul_ps(exp, cos_sin);
    return map(std::exp);
  }
  Vectorized<c10::complex<float>> exp2() const {
    // Use identity 2**x = exp(log(2) * x)
    const __m256 ln_2 = _mm256_set1_ps(c10::ln_2<float>);
    Vectorized<c10::complex<float>> scaled_values = _mm256_mul_ps(values, ln_2);
    return scaled_values.exp();
  }
  Vectorized<c10::complex<float>> expm1() const {
    return map(std::expm1);
  }
  Vectorized<c10::complex<float>> sin() const {
    return map(std::sin);
  }
  Vectorized<c10::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vectorized<c10::complex<float>> cos() const {
    return map(std::cos);
  }
  Vectorized<c10::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vectorized<c10::complex<float>> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vectorized<c10::complex<float>> floor() const {
    return _mm256_floor_ps(values);
  }
  Vectorized<c10::complex<float>> neg() const {
    auto zero = _mm256_setzero_ps();
    return _mm256_sub_ps(zero, values);
  }
  Vectorized<c10::complex<float>> round() const {
    return _mm256_round_ps(
        values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<float>> tan() const {
    return map(std::tan);
  }
  Vectorized<c10::complex<float>> tanh() const {
    return map(std::tanh);
  }
  Vectorized<c10::complex<float>> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<c10::complex<float>> reciprocal() const;
  Vectorized<c10::complex<float>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<c10::complex<float>> pow(
      const Vectorized<c10::complex<float>>& exp) const {
    __at_align__ c10::complex<float> x_tmp[size()];
    __at_align__ c10::complex<float> y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (const auto i : c10::irange(size())) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vectorized<c10::complex<float>> operator==(
      const Vectorized<c10::complex<float>>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }
  Vectorized<c10::complex<float>> operator!=(
      const Vectorized<c10::complex<float>>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
  }
  Vectorized<c10::complex<float>> operator<(
      const Vectorized<c10::complex<float>>& /*other*/) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator<=(
      const Vectorized<c10::complex<float>>& /*other*/) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator>(
      const Vectorized<c10::complex<float>>& /*other*/) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<float>> operator>=(
      const Vectorized<c10::complex<float>>& /*other*/) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<c10::complex<float>> eq(
      const Vectorized<c10::complex<float>>& other) const;
  Vectorized<c10::complex<float>> ne(
      const Vectorized<c10::complex<float>>& other) const;
};

template <>
Vectorized<c10::complex<float>> inline operator+(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  return _mm256_add_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator-(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  return _mm256_sub_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator*(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m256 sign_mask =
      _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm256_mul_ps(a, b); // ac       bd

  auto d_c = _mm256_permute_ps(b, 0xB1); // d        c
  d_c = _mm256_xor_ps(sign_mask, d_c); // d       -c
  auto ad_bc = _mm256_mul_ps(a, d_c); // ad      -bc

  auto ret = _mm256_hsub_ps(ac_bd, ad_bc); // ac - bd  ad + bc
  ret = _mm256_permute_ps(ret, 0xD8);
  return ret;
}

template <>
Vectorized<c10::complex<float>> inline operator/(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  // TODO: The vectorized implementation requires special handling for the case
  // where real number/imag number is 0/Inf/NaN.
  // //re + im*i = (a + bi)  / (c + di)
  // auto mask = _mm256_set1_ps(-0.f);
  // auto fabs_cd = _mm256_andnot_ps(mask, b);     // |c|    |d|
  // auto fabs_dc = _mm256_permute_ps(fabs_cd, 0xB1);   // |d|    |c|
  // auto scale = _mm256_rcp_ps(_mm256_max_ps(fabs_cd, fabs_dc));  // 1/sc 1/sc
  // auto a2 = _mm256_mul_ps(a, scale);         // a/sc     b/sc
  // auto b2 = _mm256_mul_ps(b, scale);         // c/sc     d/sc
  // auto acbd2 = _mm256_mul_ps(a2, b2);

  // const __m256 sign_mask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0,
  // -0.0, 0.0); auto dc2 = _mm256_permute_ps(b2, 0xB1);    // d/sc         c/sc
  // dc2 = _mm256_xor_ps(sign_mask, dc2);       // -d/|c,d|        c/sc
  // auto adbc2 = _mm256_mul_ps(a2, dc2);       //-ad/sc^2      bc/sc^2
  // auto res2 = _mm256_hadd_ps(acbd2, adbc2);  //(ac+bd)/sc^2  (bc-ad)/sc^2
  // res2 = _mm256_permute_ps(res2, 0xD8);

  // // get the denominator
  // auto denom2 = Vectorized<c10::complex<float>>(b2).abs_2_();  //
  // (c^2+d^2)/sc^2   (c^2+d^2)/sc^2 res2 = _mm256_div_ps(res2, denom2); return
  // res2;
  __at_align__ c10::complex<float>
      tmp1[Vectorized<c10::complex<float>>::size()];
  __at_align__ c10::complex<float>
      tmp2[Vectorized<c10::complex<float>>::size()];
  __at_align__ c10::complex<float> out[Vectorized<c10::complex<float>>::size()];
  a.store(tmp1);
  b.store(tmp2);
  for (const auto i : c10::irange(Vectorized<c10::complex<float>>::size())) {
    out[i] = tmp1[i] / tmp2[i];
  }
  return _mm256_loadu_ps(reinterpret_cast<const float*>(out));
}

// reciprocal. Implement this here so we can use multiplication.
inline Vectorized<c10::complex<float>> Vectorized<
    c10::complex<float>>::reciprocal() const {
  // TODO: The vectorized implementation requires special handling for the case
  // where real number/imag number is 0/Inf/NaN.
  // //re + im*i = (a + bi)  / (c + di)
  // //re = (ac + bd)/abs_2() = c/abs_2()
  // //im = (bc - ad)/abs_2() = d/abs_2()
  // const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0,
  // 0.0, -0.0); auto c_d = _mm256_xor_ps(sign_mask, values);    //c       -d
  // return _mm256_div_ps(c_d, abs_2_());
  __at_align__ c10::complex<float> tmp[size()];
  store(tmp);
  for (const auto i : c10::irange(size())) {
    tmp[i] = c10::complex<float>(1) / tmp[i];
  }
  return loadu(tmp);
}

inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::atan()
    const {
  // TODO: The vectorized implementation requires special handling for the case
  // where real number/imag number is 0/Inf/NaN.
  // // atan(x) = i/2 * ln((i + z)/(i - z))
  // const __m256 i = _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  // const Vectorized i_half = _mm256_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0,
  // 0.5);

  // auto sum = Vectorized(_mm256_add_ps(i, values));                      // a
  // 1+b auto sub = Vectorized(_mm256_sub_ps(i, values)); // -a       1-b auto
  // ln = (sum/sub).log();                                        // ln((i +
  // z)/(i - z)) return i_half*ln; // i/2*ln()
  return map(std::atan);
}

template <>
Vectorized<c10::complex<float>> inline maximum(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm256_blendv_ps(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_ps(max, isnan);
}

template <>
Vectorized<c10::complex<float>> inline minimum(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm256_blendv_ps(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_ps(min, isnan);
}

template <>
Vectorized<c10::complex<float>> inline operator&(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator|(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vectorized<c10::complex<float>> inline operator^(
    const Vectorized<c10::complex<float>>& a,
    const Vectorized<c10::complex<float>>& b) {
  return _mm256_xor_ps(a, b);
}

inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::eq(
    const Vectorized<c10::complex<float>>& other) const {
  auto eq = (*this == other); // compares real and imag individually
  // If both real numbers and imag numbers are equal, then the complex numbers
  // are equal
  return (eq.real() & eq.imag()) &
      Vectorized<c10::complex<float>>(_mm256_set1_ps(1.0f));
}

inline Vectorized<c10::complex<float>> Vectorized<c10::complex<float>>::ne(
    const Vectorized<c10::complex<float>>& other) const {
  auto ne = (*this != other); // compares real and imag individually
  // If either real numbers or imag numbers are not equal, then the complex
  // numbers are not equal
  return (ne.real() | ne.imag()) &
      Vectorized<c10::complex<float>>(_mm256_set1_ps(1.0f));
}

#endif

} // namespace CPU_CAPABILITY
} // namespace at::vec
