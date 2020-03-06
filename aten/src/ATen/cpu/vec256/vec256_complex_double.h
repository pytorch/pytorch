#pragma once

#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#if defined(__AVX__) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

template <> class Vec256<std::complex<double>> {
private:
  __m256d values;
public:
  using value_type = std::complex<double>;
  static constexpr int size() {
    return 2;
  }
  Vec256() {}
  Vec256(__m256d v) : values(v) {}
  Vec256(std::complex<double> val) {
    double real_value = val.real();
    double imag_value = val.imag();
    values = _mm256_setr_pd(real_value, imag_value,
                            real_value, imag_value);
  }
  Vec256(std::complex<double> val1, std::complex<double> val2) {
    values = _mm256_setr_pd(val1.real(), val1.imag(),
                            val2.real(), val2.imag());
  }
  operator __m256d() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<std::complex<double>> blend(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
     // convert std::complex<V> index mask to V index mask: xy -> xxyy
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm256_blend_pd(a.values, b.values, 0x03);
      case 2:
        return _mm256_blend_pd(a.values, b.values, 0x0c);
    }
    return b;
  }
  static Vec256<std::complex<double>> blendv(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                               const Vec256<std::complex<double>>& mask) {
    // convert std::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm256_unpacklo_pd(mask.values, mask.values);
    return _mm256_blendv_pd(a.values, b.values, mask_);

  }
  static Vec256<std::complex<double>> arange(std::complex<double> base = 0., std::complex<double> step = 1.) {
    return Vec256<std::complex<double>>(base,
                                        base + step);
  }
  static Vec256<std::complex<double>> set(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                            int64_t count = size()) {
    switch (count) {
      case 0:
        return a;
      case 1:
        return blend<1>(a, b);
    }
    return b;
  }
  static Vec256<std::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align32__ double tmp_values[2*size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < 2*size(); ++i) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(std::complex<double>));
    return _mm256_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm256_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      double tmp_values[2*size()];
      _mm256_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(std::complex<double>));
    }
  }
  const std::complex<double>& operator[](int idx) const  = delete;
  std::complex<double>& operator[](int idx) = delete;
  Vec256<std::complex<double>> map(std::complex<double> (*f)(const std::complex<double> &)) const {
    __at_align32__ std::complex<double> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  __m256d abs_2_() const {
    auto val_2 = _mm256_mul_pd(values, values);     // a*a     b*b
    return _mm256_hadd_pd(val_2, val_2);            // a*a+b*b a*a+b*b
  }
  __m256d abs_() const {
    return _mm256_sqrt_pd(abs_2_());                // abs     abs
  }
  Vec256<std::complex<double>> abs() const {
    const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                     0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm256_and_pd(abs_(), real_mask);        // abs     0
  }
  __m256d angle_() const {
    //angle = atan2(b/a)
    auto b_a = _mm256_permute_pd(values, 0x05);     // b        a
    return Sleef_atan2d4_u10(values, b_a);          // 90-angle angle
  }
  Vec256<std::complex<double>> angle() const {
    const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                     0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    auto angle = _mm256_permute_pd(angle_(), 0x05); // angle    90-angle
    return _mm256_and_pd(angle, real_mask);         // angle    0
  }
  __m256d real_() const {
    const __m256d real_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                     0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm256_and_pd(values, real_mask);
  }
  Vec256<std::complex<double>> real() const {
    return real_();
  }
  __m256d imag_() const {
    const __m256d imag_mask = _mm256_castsi256_pd(_mm256_setr_epi64x(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                     0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
    return _mm256_and_pd(values, imag_mask);
  }
  Vec256<std::complex<double>> imag() const {
    return _mm256_permute_pd(imag_(), 0x05);           //b        a
  }
  __m256d conj_() const {
    const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_pd(values, sign_mask);           // a       -b
  }
  Vec256<std::complex<double>> conj() const {
    return conj_();
  }
  Vec256<std::complex<double>> log() const {
    // Most trigonomic ops use the log() op to improve complex number performance.
    return map(std::log);
  }
  Vec256<std::complex<double>> log2() const {
    const __m256d log2_ = _mm256_set1_pd(std::log(2));
    return _mm256_div_pd(log(), log2_);
  }
  Vec256<std::complex<double>> log10() const {
    const __m256d log10_ = _mm256_set1_pd(std::log(10));
    return _mm256_div_pd(log(), log10_);
  }
  Vec256<std::complex<double>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m256d one = _mm256_set1_pd(1);

    auto conj = conj_();
    auto b_a = _mm256_permute_pd(conj, 0x05);                         //-b        a
    auto ab = _mm256_mul_pd(conj, b_a);                               //-ab       -ab
    auto im = _mm256_add_pd(ab, ab);                                  //-2ab      -2ab

    auto val_2 = _mm256_mul_pd(values, values);                       // a*a      b*b
    auto re = _mm256_hsub_pd(val_2, _mm256_permute_pd(val_2, 0x05));  // a*a-b*b  b*b-a*a
    re = _mm256_sub_pd(one, re);

    auto root = Vec256(_mm256_blend_pd(re, im, 0x0A)).sqrt();         //sqrt(re + i*im)
    auto ln = Vec256(_mm256_add_pd(b_a, root)).log();                 //ln(iz + sqrt())
    return Vec256(_mm256_permute_pd(ln.values, 0x05)).conj();         //-i*ln()
  }
  Vec256<std::complex<double>> acos() const {
    // acos(x) = pi/2 - asin(x)
    const __m256d pi_2 = _mm256_setr_pd(M_PI/2, 0.0, M_PI/2, 0.0);
    return _mm256_sub_pd(pi_2, asin());
  }
  Vec256<std::complex<double>> atan() const;
  Vec256<std::complex<double>> atan2(const Vec256<std::complex<double>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> exp() const {
    //exp(a + bi)
    // = exp(a)*(cos(b) + sin(b)i)
    auto exp = Sleef_expd4_u10(values);                               //exp(a)           exp(b)
    exp = _mm256_blend_pd(exp, _mm256_permute_pd(exp, 0x05), 0x0A);   //exp(a)           exp(a)

    auto sin_cos = Sleef_sincosd4_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
    auto cos_sin = _mm256_blend_pd(sin_cos.y, sin_cos.x, 0x0A);       //cos(b)           sin(b)
    return _mm256_mul_pd(exp, cos_sin);
  }
  Vec256<std::complex<double>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<double>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<double>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<double>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<double>> ceil() const {
    return _mm256_ceil_pd(values);
  }
  Vec256<std::complex<double>> floor() const {
    return _mm256_floor_pd(values);
  }
  Vec256<std::complex<double>> neg() const {
    auto zero = _mm256_setzero_pd();
    return _mm256_sub_pd(zero, values);
  }
  Vec256<std::complex<double>> round() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<std::complex<double>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<double>> tanh() const {
    return map(std::tanh);
  }
  Vec256<std::complex<double>> trunc() const {
    return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<std::complex<double>> sqrt() const {
    //   sqrt(a + bi)
    // = sqrt(2)/2 * [sqrt(sqrt(a**2 + b**2) + a) + sgn(b)*sqrt(sqrt(a**2 + b**2) - a)i]
    // = sqrt(2)/2 * [sqrt(abs() + a) + sgn(b)*sqrt(abs() - a)i]

    const __m256d scalar = _mm256_set1_pd(std::sqrt(2)/2);             //sqrt(2)/2      sqrt(2)/2
    const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    auto sign = _mm256_and_pd(values, sign_mask);
    auto factor = _mm256_or_pd(scalar, sign);

    auto a_a = _mm256_xor_pd(_mm256_movedup_pd(values), sign_mask);    // a             -a
    auto res_re_im = _mm256_sqrt_pd(_mm256_add_pd(abs_(), a_a));       // sqrt(abs + a) sqrt(abs - a)
    return _mm256_mul_pd(factor, res_re_im);
  }
  Vec256<std::complex<double>> reciprocal() const;
  Vec256<std::complex<double>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vec256<std::complex<double>> pow(const Vec256<std::complex<double>> &exp) const {
    __at_align32__ std::complex<double> x_tmp[size()];
    __at_align32__ std::complex<double> y_tmp[size()];
    store(x_tmp);
    exp.store(y_tmp);
    for (int i = 0; i < size(); i++) {
      x_tmp[i] = std::pow(x_tmp[i], y_tmp[i]);
    }
    return loadu(x_tmp);
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<std::complex<double>> operator==(const Vec256<std::complex<double>>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_EQ_OQ);
  }
  Vec256<std::complex<double>> operator!=(const Vec256<std::complex<double>>& other) const {
    return _mm256_cmp_pd(values, other.values, _CMP_NEQ_OQ);
  }
  Vec256<std::complex<double>> operator<(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> operator<=(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> operator>(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> operator>=(const Vec256<std::complex<double>>& other) const {
    AT_ERROR("not supported for complex numbers");
  }
};

template <> Vec256<std::complex<double>> inline operator+(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  return _mm256_add_pd(a, b);
}

template <> Vec256<std::complex<double>> inline operator-(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  return _mm256_sub_pd(a, b);
}

template <> Vec256<std::complex<double>> inline operator*(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm256_mul_pd(a, b);         //ac       bd

  auto d_c = _mm256_permute_pd(b, 0x05);    //d        c
  d_c = _mm256_xor_pd(sign_mask, d_c);      //d       -c
  auto ad_bc = _mm256_mul_pd(a, d_c);       //ad      -bc

  auto ret = _mm256_hsub_pd(ac_bd, ad_bc);  //ac - bd  ad + bc
  return ret;
}

template <> Vec256<std::complex<double>> inline operator/(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2()
  //im = (bc - ad)/abs_2()
  const __m256d sign_mask = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
  auto ac_bd = _mm256_mul_pd(a, b);         //ac       bd

  auto d_c = _mm256_permute_pd(b, 0x05);    //d        c
  d_c = _mm256_xor_pd(sign_mask, d_c);      //-d       c
  auto ad_bc = _mm256_mul_pd(a, d_c);       //-ad      bc

  auto re_im = _mm256_hadd_pd(ac_bd, ad_bc);//ac + bd  bc - ad
  return _mm256_div_pd(re_im, b.abs_2_());
}

// reciprocal. Implement this here so we can use multiplication.
Vec256<std::complex<double>> Vec256<std::complex<double>>::reciprocal() const{
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2() = c/abs_2()
  //im = (bc - ad)/abs_2() = d/abs_2()
  const __m256d sign_mask = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm256_xor_pd(sign_mask, values);    //c       -d
  return _mm256_div_pd(c_d, abs_2_());
}

Vec256<std::complex<double>> Vec256<std::complex<double>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  const __m256d i = _mm256_setr_pd(0.0, 1.0, 0.0, 1.0);
  const Vec256 i_half = _mm256_setr_pd(0.0, 0.5, 0.0, 0.5);

  auto sum = Vec256(_mm256_add_pd(i, values));                      // a        1+b
  auto sub = Vec256(_mm256_sub_pd(i, values));                      // -a       1-b
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vec256<std::complex<double>> inline maximum(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm256_blendv_pd(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_pd(max, isnan);
}

template <>
Vec256<std::complex<double>> inline minimum(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_pd(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm256_blendv_pd(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_pd(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_pd(min, isnan);
}

template <>
Vec256<std::complex<double>> inline clamp(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& min, const Vec256<std::complex<double>>& max) {
  auto abs_a = a.abs_2_();
  auto abs_min = min.abs_2_();
  auto max_mask = _mm256_cmp_pd(abs_a, abs_min, _CMP_LT_OQ);
  auto abs_max = max.abs_2_();
  auto min_mask = _mm256_cmp_pd(abs_a, abs_max, _CMP_GT_OQ);
  return _mm256_blendv_pd(_mm256_blendv_pd(a, min, max_mask), max, min_mask);
}

template <>
Vec256<std::complex<double>> inline clamp_min(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& min) {
  auto abs_a = a.abs_2_();
  auto abs_min = min.abs_2_();
  auto max_mask = _mm256_cmp_pd(abs_a, abs_min, _CMP_LT_OQ);
  return _mm256_blendv_pd(a, min, max_mask);
}

template <>
Vec256<std::complex<double>> inline clamp_max(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& max) {
  auto abs_a = a.abs_2_();
  auto abs_max = max.abs_2_();
  auto min_mask = _mm256_cmp_pd(abs_a, abs_max, _CMP_GT_OQ);
  return _mm256_blendv_pd(a, max, min_mask);
}

template <>
Vec256<std::complex<double>> inline operator&(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  return _mm256_and_pd(a, b);
}

template <>
Vec256<std::complex<double>> inline operator|(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  return _mm256_or_pd(a, b);
}

template <>
Vec256<std::complex<double>> inline operator^(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  return _mm256_xor_pd(a, b);
}

#ifdef __AVX2__
template <> inline Vec256<std::complex<double>> fmadd(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b, const Vec256<std::complex<double>>& c) {
  return a * b + c;
}
#endif

#endif

}}}
