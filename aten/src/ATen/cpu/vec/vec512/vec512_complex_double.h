#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <c10/util/irange.h>
#include <ATen/cpu/vec/intrinsics.h>
#include <ATen/cpu/vec/vec_base.h>
#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec {
// See Note [CPU_CAPABILITY namespace]
inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512) && !defined(_MSC_VER)

template <> class Vectorized<c10::complex<double>> {
private:
  __m512d values;
  static constexpr __m512i zero_vector {0, 0, 0, 0, 0, 0, 0, 0};
public:
  using value_type = c10::complex<double>;
  using size_type = int;
  static constexpr size_type size() {
    return 4;
  }
  Vectorized() {}
  Vectorized(__m512d v) : values(v) {}
  Vectorized(c10::complex<double> val) {
    double real_value = val.real();
    double imag_value = val.imag();
    values = _mm512_setr_pd(real_value, imag_value, real_value, imag_value,
                            real_value, imag_value, real_value, imag_value);
  }
  Vectorized(c10::complex<double> val1, c10::complex<double> val2,
            c10::complex<double> val3, c10::complex<double> val4) {
    values = _mm512_setr_pd(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag());
  }
  operator __m512d() const {
    return values;
  }
  template <int64_t mask>
  static Vectorized<c10::complex<double>> blend(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
     // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    // NOLINTNEXTLINE(clang-diagnostic-warning)
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm512_mask_blend_pd(0x03, a.values, b.values); //b0000 0001 = b0000 0011
      case 2:
        return _mm512_mask_blend_pd(0x0C, a.values, b.values); //b0000 0010 = b0000 1100
      case 3:
        return _mm512_mask_blend_pd(0x0F, a.values, b.values); //b0000 0011 = b0000 1111
      case 4:
        return _mm512_mask_blend_pd(0x30, a.values, b.values); //b0000 0100 = b0011 0000
      case 5:
        return _mm512_mask_blend_pd(0x33, a.values, b.values); //b0000 0101 = b0011 0011
      case 6:
        return _mm512_mask_blend_pd(0x3C, a.values, b.values); //b0000 0110 = b0011 1100
      case 7:
        return _mm512_mask_blend_pd(0x3F, a.values, b.values); //b0000 0111 = b0011 1111
      case 8:
        return _mm512_mask_blend_pd(0xC0, a.values, b.values); //b0000 1000 = b1100 0000
      case 9:
        return _mm512_mask_blend_pd(0xC3, a.values, b.values); //b0000 1001 = b1100 0011
      case 10:
        return _mm512_mask_blend_pd(0xCC, a.values, b.values); //b0000 1010 = b1100 1100
      case 11:
        return _mm512_mask_blend_pd(0xCF, a.values, b.values); //b0000 1011 = b1100 1111
      case 12:
        return _mm512_mask_blend_pd(0xF0, a.values, b.values); //b0000 1100 = b1111 0000
      case 13:
        return _mm512_mask_blend_pd(0xF3, a.values, b.values); //b0000 1101 = b1111 0011
      case 14:
        return _mm512_mask_blend_pd(0xFC, a.values, b.values); //b0000 1110 = b1111 1100
      case 15:
        return _mm512_mask_blend_pd(0xFF, a.values, b.values); //b0000 1111 = b1111 1111
    }
    return b;
  }
  static Vectorized<c10::complex<double>> blendv(const Vectorized<c10::complex<double>>& a,
                                                const Vectorized<c10::complex<double>>& b,
                                                const Vectorized<c10::complex<double>>& mask) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm512_unpacklo_pd(mask.values, mask.values);
    auto all_ones = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
    auto mmask = _mm512_cmp_epi64_mask(_mm512_castpd_si512(mask_), all_ones, _MM_CMPINT_EQ);
    return _mm512_mask_blend_pd(mmask, a.values, b.values);
  }
  template<typename step_t>
  static Vectorized<c10::complex<double>> arange(c10::complex<double> base = 0.,
                                                step_t step = static_cast<step_t>(1)) {
    return Vectorized<c10::complex<double>>(base,
                                           base + c10::complex<double>(1)*step,
                                           base + c10::complex<double>(2)*step,
                                           base + c10::complex<double>(3)*step);
  }
  static Vectorized<c10::complex<double>> set(const Vectorized<c10::complex<double>>& a,
                                             const Vectorized<c10::complex<double>>& b,
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
  static Vectorized<c10::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm512_loadu_pd(reinterpret_cast<const double*>(ptr));

    __at_align__ double tmp_values[2*size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (const auto i : c10::irange(2*size())) {
      tmp_values[i] = 0.0;
    }
    std::memcpy(
        tmp_values,
        reinterpret_cast<const double*>(ptr),
        count * sizeof(c10::complex<double>));
    return _mm512_load_pd(tmp_values);
  }
  void store(void* ptr, int count = size()) const {
    if (count == size()) {
      _mm512_storeu_pd(reinterpret_cast<double*>(ptr), values);
    } else if (count > 0) {
      double tmp_values[2*size()];
      _mm512_storeu_pd(reinterpret_cast<double*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<double>));
    }
  }
  const c10::complex<double>& operator[](int idx) const  = delete;
  c10::complex<double>& operator[](int idx) = delete;
  Vectorized<c10::complex<double>> map(c10::complex<double> (*const f)(const c10::complex<double> &)) const {
    __at_align__ c10::complex<double> tmp[size()];
    store(tmp);
    for (const auto i : c10::irange(size())) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  // AVX512 doesn't have horizontal add & horizontal sub instructions.
  // TODO: hadd_pd() & hsub_pd() may have scope for improvement.
  static inline __m512d hadd_pd(__m512d a, __m512d b) {
  __m512i idx1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i idx2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  return _mm512_add_pd(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                       _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
  }
  static inline __m512d hsub_pd(__m512d a, __m512d b) {
  __m512i idx1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i idx2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  return _mm512_sub_pd(_mm512_mask_permutex2var_pd(a, 0xff, idx1, b),
                       _mm512_mask_permutex2var_pd(a, 0xff, idx2, b));
  }
  __m512d abs_2_() const {
    auto val_2 = _mm512_mul_pd(values, values);     // a*a     b*b
    return hadd_pd(val_2, val_2);            // a*a+b*b a*a+b*b
  }
  __m512d abs_() const {
    auto real = _mm512_movedup_pd(values);        // real real
    // movehdup_pd does not exist...
    auto imag = _mm512_permute_pd(values, 0xff);  // imag imag
    return Sleef_hypotd8_u05(real, imag);         // abs  abs
  }
  Vectorized<c10::complex<double>> abs() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm512_and_pd(abs_(), real_mask);        // abs     0
  }
  __m512d angle_() const {
    //angle = atan2(b/a)
    auto b_a = _mm512_permute_pd(values, 0x55);     // b        a
    return Sleef_atan2d8_u10(values, b_a);          // 90-angle angle
  }
  Vectorized<c10::complex<double>> angle() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    auto angle = _mm512_permute_pd(angle_(), 0x55); // angle    90-angle
    return _mm512_and_pd(angle, real_mask);         // angle    0
  }
  Vectorized<c10::complex<double>> sgn() const {
    auto abs = abs_();
    auto zero = _mm512_setzero_pd();
    auto mask = _mm512_cmp_pd_mask(abs, zero, _CMP_EQ_OQ);
    auto div = values / abs;
    return _mm512_mask_blend_pd(mask, div, zero);
  }
  __m512d real_() const {
    const __m512d real_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000,
                                                                    0xFFFFFFFFFFFFFFFF, 0x0000000000000000));
    return _mm512_and_pd(values, real_mask);
  }
  Vectorized<c10::complex<double>> real() const {
    return real_();
  }
  __m512d imag_() const {
    const __m512d imag_mask = _mm512_castsi512_pd(_mm512_setr_epi64(0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
                                                                    0x0000000000000000, 0xFFFFFFFFFFFFFFFF));
    return _mm512_and_pd(values, imag_mask);
  }
  Vectorized<c10::complex<double>> imag() const {
    return _mm512_permute_pd(imag_(), 0x55);           //b        a
  }
  __m512d conj_() const {
    const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm512_xor_pd(values, sign_mask);           // a       -b
  }
  Vectorized<c10::complex<double>> conj() const {
    return conj_();
  }
  Vectorized<c10::complex<double>> log() const {
    // Most trigonomic ops use the log() op to improve complex number performance.
    return map(std::log);
  }
  Vectorized<c10::complex<double>> log2() const {
    const __m512d log2_ = _mm512_set1_pd(std::log(2));
    return _mm512_div_pd(log(), log2_);
  }
  Vectorized<c10::complex<double>> log10() const {
    const __m512d log10_ = _mm512_set1_pd(std::log(10));
    return _mm512_div_pd(log(), log10_);
  }
  Vectorized<c10::complex<double>> log1p() const {
    return map(std::log1p);
  }
  Vectorized<c10::complex<double>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m512d one = _mm512_set1_pd(1);

    auto conj = conj_();
    auto b_a = _mm512_permute_pd(conj, 0x55);                         //-b        a
    auto ab = _mm512_mul_pd(conj, b_a);                               //-ab       -ab
    auto im = _mm512_add_pd(ab, ab);                                  //-2ab      -2ab

    auto val_2 = _mm512_mul_pd(values, values);                       // a*a      b*b
    auto re = hsub_pd(val_2, _mm512_permute_pd(val_2, 0x55));  // a*a-b*b  b*b-a*a
    re = _mm512_sub_pd(one, re);

    auto root = Vectorized(_mm512_mask_blend_pd(0xAA, re, im)).sqrt();         //sqrt(re + i*im)
    auto ln = Vectorized(_mm512_add_pd(b_a, root)).log();                 //ln(iz + sqrt())
    return Vectorized(_mm512_permute_pd(ln.values, 0x55)).conj();         //-i*ln()
  }
  Vectorized<c10::complex<double>> acos() const {
    // acos(x) = pi/2 - asin(x)
    constexpr auto pi_2d = c10::pi<double> / 2;
    const __m512d pi_2 = _mm512_setr_pd(pi_2d, 0.0, pi_2d, 0.0, pi_2d, 0.0, pi_2d, 0.0);
    return _mm512_sub_pd(pi_2, asin());
  }
  Vectorized<c10::complex<double>> acosh() const {
    return map(std::acosh);
  }
  Vectorized<c10::complex<double>> atan() const;
  Vectorized<c10::complex<double>> atan2(const Vectorized<c10::complex<double>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> exp() const {
    //exp(a + bi)
    // = exp(a)*(cos(b) + sin(b)i)
    auto exp = Sleef_expd8_u10(values);                               //exp(a)           exp(b)
    exp = _mm512_mask_blend_pd(0xAA, exp, _mm512_permute_pd(exp, 0x55));   //exp(a)           exp(a)

    auto sin_cos = Sleef_sincosd8_u10(values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
    auto cos_sin = _mm512_mask_blend_pd(0xAA, _mm512_permute_pd(sin_cos.y, 0x55),
                                   sin_cos.x);                  //cos(b)           sin(b)
    return _mm512_mul_pd(exp, cos_sin);
  }
  Vectorized<c10::complex<double>> exp2() const {
    // Use identity 2**x = exp(log(2) * x)
    const __m512d ln_2 = _mm512_set1_pd(c10::ln_2<double>);
    Vectorized<c10::complex<double>> scaled_values = _mm512_mul_pd(values, ln_2);
    return scaled_values.exp();
  }
  Vectorized<c10::complex<double>> expm1() const {
    return map(std::expm1);
  }
  Vectorized<c10::complex<double>> sin() const {
    return map(std::sin);
  }
  Vectorized<c10::complex<double>> sinh() const {
    return map(std::sinh);
  }
  Vectorized<c10::complex<double>> cos() const {
    return map(std::cos);
  }
  Vectorized<c10::complex<double>> cosh() const {
    return map(std::cosh);
  }
  Vectorized<c10::complex<double>> ceil() const {
    return _mm512_ceil_pd(values);
  }
  Vectorized<c10::complex<double>> floor() const {
    return _mm512_floor_pd(values);
  }
  Vectorized<c10::complex<double>> hypot(const Vectorized<c10::complex<double>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> igamma(const Vectorized<c10::complex<double>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> igammac(const Vectorized<c10::complex<double>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> neg() const {
    auto zero = _mm512_setzero_pd();
    return _mm512_sub_pd(zero, values);
  }
  Vectorized<c10::complex<double>> nextafter(const Vectorized<c10::complex<double>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> round() const {
    return _mm512_roundscale_pd(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<double>> tan() const {
    return map(std::tan);
  }
  Vectorized<c10::complex<double>> tanh() const {
    return map(std::tanh);
  }
  Vectorized<c10::complex<double>> trunc() const {
    return _mm512_roundscale_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vectorized<c10::complex<double>> sqrt() const {
    return map(std::sqrt);
  }
  Vectorized<c10::complex<double>> reciprocal() const;
  Vectorized<c10::complex<double>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vectorized<c10::complex<double>> pow(const Vectorized<c10::complex<double>> &exp) const {
    __at_align__ c10::complex<double> x_tmp[size()];
    __at_align__ c10::complex<double> y_tmp[size()];
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
  Vectorized<c10::complex<double>> operator==(const Vectorized<c10::complex<double>>& other) const {
    auto mask = _mm512_cmp_pd_mask(values, other.values, _CMP_EQ_OQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }
  Vectorized<c10::complex<double>> operator!=(const Vectorized<c10::complex<double>>& other) const {
    auto mask = _mm512_cmp_pd_mask(values, other.values, _CMP_NEQ_UQ);
    return _mm512_castsi512_pd(_mm512_mask_set1_epi64(zero_vector, mask,
                                                      0xFFFFFFFFFFFFFFFF));
  }
  Vectorized<c10::complex<double>> operator<(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> operator<=(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> operator>(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> operator>=(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vectorized<c10::complex<double>> eq(const Vectorized<c10::complex<double>>& other) const;
  Vectorized<c10::complex<double>> ne(const Vectorized<c10::complex<double>>& other) const;
  Vectorized<c10::complex<double>> lt(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> le(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> gt(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vectorized<c10::complex<double>> ge(const Vectorized<c10::complex<double>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

template <> Vectorized<c10::complex<double>> inline operator+(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  return _mm512_add_pd(a, b);
}

template <> Vectorized<c10::complex<double>> inline operator-(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  return _mm512_sub_pd(a, b);
}

template <> Vectorized<c10::complex<double>> inline operator*(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm512_mul_pd(a, b);         //ac       bd

  auto d_c = _mm512_permute_pd(b, 0x55);    //d        c
  d_c = _mm512_xor_pd(sign_mask, d_c);      //d       -c
  auto ad_bc = _mm512_mul_pd(a, d_c);       //ad      -bc

  auto ret = Vectorized<c10::complex<double>>::hsub_pd(ac_bd, ad_bc);  //ac - bd  ad + bc
  return ret;
}

template <> Vectorized<c10::complex<double>> inline operator/(const Vectorized<c10::complex<double>> &a,
                                                             const Vectorized<c10::complex<double>> &b) {
  //re + im*i = (a + bi)  / (c + di)
  auto mask = _mm512_set1_pd(-0.f);
  auto fabs_cd = _mm512_andnot_pd(mask, b);     // |c|    |d|
  auto fabs_dc = _mm512_permute_pd(fabs_cd, 0x55);   // |d|    |c|
  auto scale = _mm512_rcp14_pd(_mm512_max_pd(fabs_cd, fabs_dc));  // 1/sc     1/sc
  auto a2 = _mm512_mul_pd(a, scale);         // a/sc     b/sc
  auto b2 = _mm512_mul_pd(b, scale);         // c/sc     d/sc
  auto acbd2 = _mm512_mul_pd(a2, b2);

  const __m512d sign_mask = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto dc2 = _mm512_permute_pd(b2, 0x55);    // d/sc         c/sc
  dc2 = _mm512_xor_pd(sign_mask, dc2);       // -d/|c,d|        c/sc
  auto adbc2 = _mm512_mul_pd(a2, dc2);       //-ad/sc^2      bc/sc^2
  auto res2 = Vectorized<c10::complex<double>>::hadd_pd(acbd2, adbc2);  //(ac+bd)/sc^2  (bc-ad)/sc^2

  // get the denominator
  auto denom2 = Vectorized<c10::complex<double>>(b2).abs_2_();  // (c^2+d^2)/sc^2   (c^2+d^2)/sc^2
  res2 = _mm512_div_pd(res2, denom2);
  return res2;
}

// reciprocal. Implement this here so we can use multiplication.
inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::reciprocal() const{
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2() = c/abs_2()
  //im = (bc - ad)/abs_2() = d/abs_2()
  const __m512d sign_mask = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm512_xor_pd(sign_mask, values);    //c       -d
  return _mm512_div_pd(c_d, abs_2_());
}

inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  const __m512d i = _mm512_setr_pd(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  const Vectorized i_half = _mm512_setr_pd(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  auto sum = Vectorized(_mm512_add_pd(i, values));                      // a        1+b
  auto sub = Vectorized(_mm512_sub_pd(i, values));                      // -a       1-b
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vectorized<c10::complex<double>> inline maximum(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm512_mask_blend_pd(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                      0xFFFFFFFFFFFFFFFF);
  return _mm512_or_pd(max, _mm512_castsi512_pd(isnan));
}

template <>
Vectorized<c10::complex<double>> inline minimum(const Vectorized<c10::complex<double>>& a,
                                               const Vectorized<c10::complex<double>>& b) {
  auto zero_vec = _mm512_set1_epi64(0);
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm512_mask_blend_pd(mask, a, b);
  // Exploit the fact that all-ones is a NaN.
  auto isnan_mask = _mm512_cmp_pd_mask(abs_a, abs_b, _CMP_UNORD_Q);
  auto isnan = _mm512_mask_set1_epi64(zero_vec, isnan_mask,
                                      0xFFFFFFFFFFFFFFFF);
  return _mm512_or_pd(min, _mm512_castsi512_pd(isnan));
}

template <>
Vectorized<c10::complex<double>> inline operator&(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  return _mm512_and_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator|(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  return _mm512_or_pd(a, b);
}

template <>
Vectorized<c10::complex<double>> inline operator^(const Vectorized<c10::complex<double>>& a,
                                                 const Vectorized<c10::complex<double>>& b) {
  return _mm512_xor_pd(a, b);
}

inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::eq(const Vectorized<c10::complex<double>>& other) const {
  auto eq = (*this == other);  // compares real and imag individually
  // If both real numbers and imag numbers are equal, then the complex numbers are equal
  return (eq.real() & eq.imag()) & Vectorized<c10::complex<double>>(_mm512_set1_pd(1.0));
}

inline Vectorized<c10::complex<double>> Vectorized<c10::complex<double>>::ne(const Vectorized<c10::complex<double>>& other) const {
  auto ne = (*this != other);  // compares real and imag individually
  // If either real numbers or imag numbers are not equal, then the complex numbers are not equal
  return (ne.real() | ne.imag()) & Vectorized<c10::complex<double>>(_mm512_set1_pd(1.0));
}

#endif

}}}
