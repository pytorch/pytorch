#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
// See Note [Do not compile initializers with AVX]

#include <c10/util/complex.h>
#include <ATen/cpu/vec256/intrinsics.h>
#include <ATen/cpu/vec256/vec256_base.h>
#if (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
// See Note [Acceptable use of anonymous namespace in header]
namespace {

#if (defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2)) && !defined(_MSC_VER)

template <> class Vec256<c10::complex<float>> {
private:
  __m256 values;
public:
  using value_type = c10::complex<float>;
  static constexpr int size() {
    return 4;
  }
  Vec256() {}
  Vec256(__m256 v) : values(v) {}
  Vec256(c10::complex<float> val) {
    float real_value = val.real();
    float imag_value = val.imag();
    values = _mm256_setr_ps(real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value,
                            real_value, imag_value
                            );
  }
  Vec256(c10::complex<float> val1, c10::complex<float> val2, c10::complex<float> val3, c10::complex<float> val4) {
    values = _mm256_setr_ps(val1.real(), val1.imag(),
                            val2.real(), val2.imag(),
                            val3.real(), val3.imag(),
                            val4.real(), val4.imag()
                            );
  }
  operator __m256() const {
    return values;
  }
  template <int64_t mask>
  static Vec256<c10::complex<float>> blend(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
     // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    switch (mask) {
      case 0:
        return a;
      case 1:
        return _mm256_blend_ps(a.values, b.values, 0x03); //b0000 0001 = b0000 0011
      case 2:
        return _mm256_blend_ps(a.values, b.values, 0x0C); //b0000 0010 = b0000 1100
      case 3:
        return _mm256_blend_ps(a.values, b.values, 0x0F); //b0000 0011 = b0000 1111
      case 4:
        return _mm256_blend_ps(a.values, b.values, 0x30); //b0000 0100 = b0011 0000
      case 5:
        return _mm256_blend_ps(a.values, b.values, 0x33); //b0000 0101 = b0011 0011
      case 6:
        return _mm256_blend_ps(a.values, b.values, 0x3C); //b0000 0110 = b0011 1100
      case 7:
        return _mm256_blend_ps(a.values, b.values, 0x3F); //b0000 0111 = b0011 1111
      case 8:
        return _mm256_blend_ps(a.values, b.values, 0xC0); //b0000 1000 = b1100 0000
      case 9:
        return _mm256_blend_ps(a.values, b.values, 0xC3); //b0000 1001 = b1100 0011
      case 10:
        return _mm256_blend_ps(a.values, b.values, 0xCC); //b0000 1010 = b1100 1100
      case 11:
        return _mm256_blend_ps(a.values, b.values, 0xCF); //b0000 1011 = b1100 1111
      case 12:
        return _mm256_blend_ps(a.values, b.values, 0xF0); //b0000 1100 = b1111 0000
      case 13:
        return _mm256_blend_ps(a.values, b.values, 0xF3); //b0000 1101 = b1111 0011
      case 14:
        return _mm256_blend_ps(a.values, b.values, 0xFC); //b0000 1110 = b1111 1100
    }
    return b;
  }
  static Vec256<c10::complex<float>> blendv(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b,
                               const Vec256<c10::complex<float>>& mask) {
    // convert c10::complex<V> index mask to V index mask: xy -> xxyy
    auto mask_ = _mm256_unpacklo_ps(mask.values, mask.values);
    return _mm256_blendv_ps(a.values, b.values, mask_);

  }
  template<typename step_t>
  static Vec256<c10::complex<float>> arange(c10::complex<float> base = 0., step_t step = static_cast<step_t>(1)) {
    return Vec256<c10::complex<float>>(base,
                                        base + step,
                                        base + c10::complex<float>(2)*step,
                                        base + c10::complex<float>(3)*step);
  }
  static Vec256<c10::complex<float>> set(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b,
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
  static Vec256<c10::complex<float>> loadu(const void* ptr, int64_t count = size()) {
    if (count == size())
      return _mm256_loadu_ps(reinterpret_cast<const float*>(ptr));

    __at_align32__ float tmp_values[2*size()];
    // Ensure uninitialized memory does not change the output value See https://github.com/pytorch/pytorch/issues/32502
    // for more details. We do not initialize arrays to zero using "={0}" because gcc would compile it to two
    // instructions while a loop would be compiled to one instruction.
    for (auto i = 0; i < 2*size(); ++i) {
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
      float tmp_values[2*size()];
      _mm256_storeu_ps(reinterpret_cast<float*>(tmp_values), values);
      std::memcpy(ptr, tmp_values, count * sizeof(c10::complex<float>));
    }
  }
  const c10::complex<float>& operator[](int idx) const  = delete;
  c10::complex<float>& operator[](int idx) = delete;
  Vec256<c10::complex<float>> map(c10::complex<float> (*f)(const c10::complex<float> &)) const {
    __at_align32__ c10::complex<float> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  __m256 abs_2_() const {
    auto val_2 = _mm256_mul_ps(values, values);     // a*a     b*b
    auto ret = _mm256_hadd_ps(val_2, val_2);        // a*a+b*b a*a+b*b
    return _mm256_permute_ps(ret, 0xD8);
  }
  __m256 abs_() const {
    return _mm256_sqrt_ps(abs_2_());                // abs     abs
  }
  Vec256<c10::complex<float>> abs() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm256_and_ps(abs_(), real_mask);        // abs     0
  }
  __m256 angle_() const {
    //angle = atan2(b/a)
    auto b_a = _mm256_permute_ps(values, 0xB1);     // b        a
    return Sleef_atan2f8_u10(values, b_a);          // 90-angle angle
  }
  Vec256<c10::complex<float>> angle() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    auto angle = _mm256_permute_ps(angle_(), 0xB1); // angle    90-angle
    return _mm256_and_ps(angle, real_mask);         // angle    0
  }
  Vec256<c10::complex<float>> sgn() const {
    auto abs = abs_();
    auto zero = _mm256_setzero_ps();
    auto mask = _mm256_cmp_ps(abs, zero, _CMP_EQ_OQ);
    auto abs_val = Vec256(abs);

    auto div = values / abs_val.values;       // x / abs(x)

    return _mm256_blendv_ps(div, zero, mask);
  }
  __m256 real_() const {
    const __m256 real_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000,
                                                                   0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000));
    return _mm256_and_ps(values, real_mask);
  }
  Vec256<c10::complex<float>> real() const {
    return real_();
  }
  __m256 imag_() const {
    const __m256 imag_mask = _mm256_castsi256_ps(_mm256_setr_epi32(0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF,
                                                                   0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF));
    return _mm256_and_ps(values, imag_mask);
  }
  Vec256<c10::complex<float>> imag() const {
    return _mm256_permute_ps(imag_(), 0xB1);        //b        a
  }
  __m256 conj_() const {
    const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    return _mm256_xor_ps(values, sign_mask);        // a       -b
  }
  Vec256<c10::complex<float>> conj() const {
    return conj_();
  }
  Vec256<c10::complex<float>> log() const {
    // Most trigonomic ops use the log() op to improve complex number performance.
    return map(std::log);
  }
  Vec256<c10::complex<float>> log2() const {
    const __m256 log2_ = _mm256_set1_ps(std::log(2));
    return _mm256_div_ps(log(), log2_);
  }
  Vec256<c10::complex<float>> log10() const {
    const __m256 log10_ = _mm256_set1_ps(std::log(10));
    return _mm256_div_ps(log(), log10_);
  }
  Vec256<c10::complex<float>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> asin() const {
    // asin(x)
    // = -i*ln(iz + sqrt(1 -z^2))
    // = -i*ln((ai - b) + sqrt(1 - (a + bi)*(a + bi)))
    // = -i*ln((-b + ai) + sqrt(1 - (a**2 - b**2) - 2*abi))
    const __m256 one = _mm256_set1_ps(1);

    auto conj = conj_();
    auto b_a = _mm256_permute_ps(conj, 0xB1);                         //-b        a
    auto ab = _mm256_mul_ps(conj, b_a);                               //-ab       -ab
    auto im = _mm256_add_ps(ab, ab);                                  //-2ab      -2ab

    auto val_2 = _mm256_mul_ps(values, values);                       // a*a      b*b
    auto re = _mm256_hsub_ps(val_2, _mm256_permute_ps(val_2, 0xB1));  // a*a-b*b  b*b-a*a
    re = _mm256_permute_ps(re, 0xD8);
    re = _mm256_sub_ps(one, re);

    auto root = Vec256(_mm256_blend_ps(re, im, 0xAA)).sqrt();         //sqrt(re + i*im)
    auto ln = Vec256(_mm256_add_ps(b_a, root)).log();                 //ln(iz + sqrt())
    return Vec256(_mm256_permute_ps(ln.values, 0xB1)).conj();         //-i*ln()
  }
  Vec256<c10::complex<float>> acos() const {
    // acos(x) = pi/2 - asin(x)
    const __m256 pi_2 = _mm256_setr_ps(M_PI/2, 0.0, M_PI/2, 0.0, M_PI/2, 0.0, M_PI/2, 0.0);
    return _mm256_sub_ps(pi_2, asin());
  }
  Vec256<c10::complex<float>> atan() const;
  Vec256<c10::complex<float>> atan2(const Vec256<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> erf() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> erfc() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> exp() const {
    // Non Vectorized Code for Reference
    // Link:
    // https://github.com/llvm/llvm-project/blob/5e9e335a247040a175855f99dbab5957064434ba/libcxx/include/complex#L1054-L1077
    // _Tp __i = __x.imag();
    // if (std::isinf(__x.real()))
    // {
    //     if (__x.real() < _Tp(0)) // Step 1
    //     {
    //         if (!std::isfinite(__i))
    //             __i = _Tp(1);
    //     }
    //     else if (__i == 0 || !std::isfinite(__i)) // Step 2
    //     {
    //         if (std::isinf(__i)) // Step 2.1
    //             __i = _Tp(NAN);
    //         return std::complex<_Tp>(__x.real(), __i);
    //     }
    // }
    // else if (std::isnan(__x.real()) && __x.imag() == 0) // Step 3
    //     return __x;
    // _Tp __e = exp(__x.real());
    // return std::complex<_Tp>(__e * cos(__i), __e * sin(__i));

    auto pos_inf = __m256{INFINITY,INFINITY, INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    auto neg_inf = __m256{-INFINITY,-INFINITY, -INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    auto nan_vec = __m256{NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN};
    auto one_vec = __m256{1, 1, 1, 1, 1, 1, 1, 1};
    auto zero_vec = _mm256_setzero_ps();
    auto real_mask = __m256{-NAN, 0, -NAN, 0, -NAN, 0, -NAN, 0};
    auto imag_mask = __m256{0, -NAN, 0, -NAN, 0, -NAN, 0, -NAN};

    auto zero_mask = _mm256_cmp_ps(values, zero_vec, _CMP_EQ_OQ);
    auto pos_inf_mask = _mm256_cmp_ps(values, pos_inf, _CMP_EQ_OQ);
    auto neg_inf_mask = _mm256_cmp_ps(values, neg_inf, _CMP_EQ_OQ);
    auto is_inf_mask = _mm256_or_ps(pos_inf_mask, neg_inf_mask);
    auto not_nan_mask = _mm256_cmp_ps(values, values, _CMP_EQ_OQ);
    auto nan_mask = _mm256_cmp_ps(not_nan_mask, zero_vec, _CMP_EQ_OQ);
    auto not_is_finite_mask = _mm256_or_ps(is_inf_mask, nan_mask);

    // if real is neginf and imag is not finite [Step 1]
    auto neg_inf_real_mask = _mm256_permute_ps(neg_inf_mask, 0xB1);
    auto real_neg_inf_imag_not_finite = _mm256_and_ps(not_is_finite_mask, neg_inf_real_mask);
    auto real_neg_inf_imag_not_finite_mask = _mm256_and_ps(real_neg_inf_imag_not_finite, imag_mask);
    auto updated_values = _mm256_blendv_ps(values, one_vec, real_neg_inf_imag_not_finite_mask);

    // if real is pos_inf and imag is 0 or not-finite [Step 2]
    auto pos_inf_real_mask = _mm256_permute_ps(pos_inf_mask, 0xB1);

    auto zero_or_not_finite = _mm256_or_ps(zero_mask, not_is_finite_mask);
    auto zero_or_not_finite_imag = _mm256_and_ps(zero_or_not_finite, imag_mask);
    auto pos_inf_real_zero_or_not_finite_imag =
        _mm256_and_ps(zero_or_not_finite_imag, pos_inf_real_mask);

    // if real is pos_inf and imag is inf [Step 2.1]
    auto infinity_imag = _mm256_and_ps(is_inf_mask, imag_mask);
    auto pos_inf_real_zero_or_infinity_imag =
        _mm256_and_ps(infinity_imag, pos_inf_real_mask);
    auto pos_inf_real_not_finite_imag =
        _mm256_blendv_ps(values, nan_vec, pos_inf_real_zero_or_infinity_imag);
    auto pos_inf_real_not_finite_imag_mask = _mm256_blend_ps(
        pos_inf_real_zero_or_not_finite_imag,
        _mm256_permute_ps(pos_inf_real_zero_or_not_finite_imag, 0xB1),
        0x55);

    //(std::isnan(__x.real()) && __x.imag() == 0) [Step 3]
    auto real_is_nan = _mm256_and_ps(nan_mask, real_mask);
    auto imag_is_zero = _mm256_and_ps(zero_mask, imag_mask);
    auto imag_is_zero_shift = _mm256_permute_ps(imag_is_zero, 0xB1);
    auto real_is_nan_imag_zero_mask = _mm256_and_ps(real_is_nan, imag_is_zero_shift);
    real_is_nan_imag_zero_mask = _mm256_blend_ps(
        real_is_nan_imag_zero_mask,
        _mm256_permute_ps(real_is_nan_imag_zero_mask, 0xB1),
        0xAA);

    // Exp MUL
    auto exp = Sleef_expf8_u10(updated_values);                               //exp(a)           exp(b)
    exp = _mm256_blend_ps(exp, _mm256_permute_ps(exp, 0xB1), 0xAA);   //exp(a)           exp(a)

    auto sin_cos = Sleef_sincosf8_u10(updated_values);                        //[sin(a), cos(a)] [sin(b), cos(b)]
    auto cos_sin = _mm256_blend_ps(_mm256_permute_ps(sin_cos.y, 0xB1),
                                   sin_cos.x, 0xAA);                  //cos(b)           sin(b)

    auto exp_computed = _mm256_mul_ps(exp, cos_sin);

    // Handle inf in computation.
    auto computed_pos_inf_mask = _mm256_cmp_ps(exp, pos_inf, _CMP_EQ_OQ);
    auto computed_pos_inf_real_mask_exp = _mm256_permute_ps(computed_pos_inf_mask, 0xB1);
    auto computed_zero_mask = _mm256_and_ps(zero_mask, imag_mask);
    auto pos_inf_real_zero_imag_exp = _mm256_and_ps(computed_zero_mask, computed_pos_inf_real_mask_exp);
    exp_computed = _mm256_blendv_ps(exp_computed, values, pos_inf_real_zero_imag_exp);

    // Handle previously computed extremal values [Step 1, Step 2]
    exp_computed = _mm256_blendv_ps(
        exp_computed,
        pos_inf_real_not_finite_imag,
        pos_inf_real_not_finite_imag_mask);
    exp_computed = _mm256_blendv_ps(exp_computed, values, real_is_nan_imag_zero_mask);
    return exp_computed;
  }
  Vec256<c10::complex<float>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> sin() const {
    return map(std::sin);
  }
  Vec256<c10::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vec256<c10::complex<float>> cos() const {
    return map(std::cos);
  }
  Vec256<c10::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vec256<c10::complex<float>> ceil() const {
    return _mm256_ceil_ps(values);
  }
  Vec256<c10::complex<float>> floor() const {
    return _mm256_floor_ps(values);
  }
  Vec256<c10::complex<float>> hypot(const Vec256<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> igamma(const Vec256<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> igammac(const Vec256<c10::complex<float>> &x) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> neg() const {
    auto zero = _mm256_setzero_ps();
    return _mm256_sub_ps(zero, values);
  }
  Vec256<c10::complex<float>> nextafter(const Vec256<c10::complex<float>> &b) const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<c10::complex<float>> round() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  }
  Vec256<c10::complex<float>> tan() const {
    return map(std::tan);
  }
  Vec256<c10::complex<float>> tanh() const {
    return map(std::tanh);
  }
  Vec256<c10::complex<float>> trunc() const {
    return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  }
  Vec256<c10::complex<float>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<c10::complex<float>> reciprocal() const;
  Vec256<c10::complex<float>> rsqrt() const {
    return sqrt().reciprocal();
  }
  Vec256<c10::complex<float>> pow(const Vec256<c10::complex<float>> &exp) const {
    __at_align32__ c10::complex<float> x_tmp[size()];
    __at_align32__ c10::complex<float> y_tmp[size()];
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
  Vec256<c10::complex<float>> operator==(const Vec256<c10::complex<float>>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_EQ_OQ);
  }
  Vec256<c10::complex<float>> operator!=(const Vec256<c10::complex<float>>& other) const {
    return _mm256_cmp_ps(values, other.values, _CMP_NEQ_UQ);
  }
  Vec256<c10::complex<float>> operator<(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> operator<=(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> operator>(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> operator>=(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }

  Vec256<c10::complex<float>> eq(const Vec256<c10::complex<float>>& other) const;
  Vec256<c10::complex<float>> ne(const Vec256<c10::complex<float>>& other) const;
  Vec256<c10::complex<float>> lt(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> le(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> gt(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
  Vec256<c10::complex<float>> ge(const Vec256<c10::complex<float>>& other) const {
    TORCH_CHECK(false, "not supported for complex numbers");
  }
};

template <> Vec256<c10::complex<float>> inline operator+(const Vec256<c10::complex<float>> &a, const Vec256<c10::complex<float>> &b) {
  return _mm256_add_ps(a, b);
}

template <> Vec256<c10::complex<float>> inline operator-(const Vec256<c10::complex<float>> &a, const Vec256<c10::complex<float>> &b) {
  return _mm256_sub_ps(a, b);
}

template <> Vec256<c10::complex<float>> inline operator*(const Vec256<c10::complex<float>> &a, const Vec256<c10::complex<float>> &b) {
  //(a + bi)  * (c + di) = (ac - bd) + (ad + bc)i
  const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto ac_bd = _mm256_mul_ps(a, b);         //ac       bd

  auto d_c = _mm256_permute_ps(b, 0xB1);    //d        c
  d_c = _mm256_xor_ps(sign_mask, d_c);      //d       -c
  auto ad_bc = _mm256_mul_ps(a, d_c);       //ad      -bc

  auto ret = _mm256_hsub_ps(ac_bd, ad_bc);  //ac - bd  ad + bc
  ret = _mm256_permute_ps(ret, 0xD8);
  return ret;
}

template <> Vec256<c10::complex<float>> inline operator/(const Vec256<c10::complex<float>> &a, const Vec256<c10::complex<float>> &b) {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2()
  //im = (bc - ad)/abs_2()
  const __m256 sign_mask = _mm256_setr_ps(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
  auto ac_bd = _mm256_mul_ps(a, b);         //ac       bd

  auto d_c = _mm256_permute_ps(b, 0xB1);    //d        c
  d_c = _mm256_xor_ps(sign_mask, d_c);      //-d       c
  auto ad_bc = _mm256_mul_ps(a, d_c);       //-ad      bc

  auto re_im = _mm256_hadd_ps(ac_bd, ad_bc);//ac + bd  bc - ad
  re_im = _mm256_permute_ps(re_im, 0xD8);
  return _mm256_div_ps(re_im, b.abs_2_());
}

// reciprocal. Implement this here so we can use multiplication.
Vec256<c10::complex<float>> Vec256<c10::complex<float>>::reciprocal() const {
  //re + im*i = (a + bi)  / (c + di)
  //re = (ac + bd)/abs_2() = c/abs_2()
  //im = (bc - ad)/abs_2() = d/abs_2()
  const __m256 sign_mask = _mm256_setr_ps(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
  auto c_d = _mm256_xor_ps(sign_mask, values);    //c       -d
  return _mm256_div_ps(c_d, abs_2_());
}

Vec256<c10::complex<float>> Vec256<c10::complex<float>>::atan() const {
  // atan(x) = i/2 * ln((i + z)/(i - z))
  const __m256 i = _mm256_setr_ps(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
  const Vec256 i_half = _mm256_setr_ps(0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5);

  auto sum = Vec256(_mm256_add_ps(i, values));                      // a        1+b
  auto sub = Vec256(_mm256_sub_ps(i, values));                      // -a       1-b
  auto ln = (sum/sub).log();                                        // ln((i + z)/(i - z))
  return i_half*ln;                                                 // i/2*ln()
}

template <>
Vec256<c10::complex<float>> inline maximum(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_LT_OQ);
  auto max = _mm256_blendv_ps(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_ps(max, isnan);
}

template <>
Vec256<c10::complex<float>> inline minimum(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
  auto abs_a = a.abs_2_();
  auto abs_b = b.abs_2_();
  auto mask = _mm256_cmp_ps(abs_a, abs_b, _CMP_GT_OQ);
  auto min = _mm256_blendv_ps(a, b, mask);
  // Exploit the fact that all-ones is a NaN.
  auto isnan = _mm256_cmp_ps(abs_a, abs_b, _CMP_UNORD_Q);
  return _mm256_or_ps(min, isnan);
}

template <>
Vec256<c10::complex<float>> inline operator&(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
  return _mm256_and_ps(a, b);
}

template <>
Vec256<c10::complex<float>> inline operator|(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
  return _mm256_or_ps(a, b);
}

template <>
Vec256<c10::complex<float>> inline operator^(const Vec256<c10::complex<float>>& a, const Vec256<c10::complex<float>>& b) {
  return _mm256_xor_ps(a, b);
}

Vec256<c10::complex<float>> Vec256<c10::complex<float>>::eq(
    const Vec256<c10::complex<float>>& other) const {
  return (*this == other) & Vec256<c10::complex<float>>(_mm256_set1_ps(1.0f));
}

Vec256<c10::complex<float>> Vec256<c10::complex<float>>::ne(
    const Vec256<c10::complex<float>>& other) const {
  return (*this != other) & Vec256<c10::complex<float>>(_mm256_set1_ps(1.0f));
}

#endif

}}}
