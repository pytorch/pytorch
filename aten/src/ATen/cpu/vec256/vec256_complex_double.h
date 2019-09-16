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
  __m256d real_values;
  __m256d imag_values;
public:
  using value_type = std::complex<double>;
  static constexpr int size() {
    return 4;
  }
  Vec256() {}
  Vec256(__m256d real) : real_values(real), imag_values(_mm256_setzero_pd()) {}
  Vec256(__m256d real, __m256d imag) : real_values(real), imag_values(imag) {}
  Vec256(std::complex<double> val) {
    real_values = _mm256_set1_pd(val.real());
    imag_values = _mm256_set1_pd(val.imag());
  }
  Vec256(std::complex<double> val1, std::complex<double> val2, std::complex<double> val3, std::complex<double> val4) {
    real_values = _mm256_setr_pd(val1.real(), val2.real(), val3.real(), val4.real());
    imag_values = _mm256_setr_pd(val1.imag(), val2.imag(), val3.imag(), val4.imag());\
  }
  operator __m256d() const {
    return real_values;
  }
  template <int64_t mask>
  static Vec256<std::complex<double>> blend(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
    return Vec256<std::complex<double>>(
      _mm256_blend_pd(a.real_values, b.imag_values, mask),
      _mm256_blend_pd(b.real_values, b.imag_values, mask)
    );
  }
  static Vec256<std::complex<double>> blendv(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
                               const Vec256<std::complex<double>>& mask) {
    return Vec256<std::complex<double>>(
      _mm256_blendv_pd(a.real_values, b.real_values, mask.real_values),
      _mm256_blendv_pd(a.imag_values, b.imag_values, mask.imag_values)
    );
  }
  static Vec256<std::complex<double>> arange(std::complex<double> base = 0., std::complex<double> step = 1.) {
    return Vec256<std::complex<double>>(base, base + step, base + 2.0 * step, base + 3.0 * step);
  }
  static Vec256<std::complex<double>> set(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b,
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
  static Vec256<std::complex<double>> loadu(const void* ptr, int64_t count = size()) {
    __at_align32__ double tmp_reals[size()];
    __at_align32__ double tmp_imags[size()];

    const double * tmp = reinterpret_cast<const double *>(ptr);
    for (int i=0; i<count; i++)
    {
      tmp_reals[i] = tmp[2*i];
      tmp_imags[i] = tmp[2*i+1];
    }
    return Vec256<std::complex<double>>(_mm256_loadu_pd(tmp_reals), _mm256_loadu_pd(tmp_imags));
  }
  void store(void* ptr, int count = size()) const {
    double tmp[2*size()];
    _mm256_storeu_pd(tmp, real_values);
    _mm256_storeu_pd(tmp + size(), imag_values);

    // permute to correct order
    double *tmp_ptr = reinterpret_cast<double *>(ptr);
    for (int i=0; i<count; i++)
    {
      tmp_ptr[2*i] = tmp[i];
      tmp_ptr[2*i+1] = tmp[i + size()];
    }
  }
  const std::complex<double>& operator[](int idx) const  = delete;
  std::complex<double>& operator[](int idx) = delete;
  Vec256<std::complex<double>> map(std::complex<double> (*f)(std::complex<double>)) const {
    __at_align32__ std::complex<double> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<std::complex<double>> map(std::complex<double> (*f)(const std::complex<double> &)) const {
    __at_align32__ std::complex<double> tmp[size()];
    store(tmp);
    for (int i = 0; i < size(); i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }
  Vec256<std::complex<double>> abs() const {
   auto real = _mm256_mul_pd(real_values, real_values);
   auto imag = _mm256_mul_pd(imag_values, imag_values);
   auto abs_2 = _mm256_add_pd(real, imag);
   return Vec256<std::complex<double>>(_mm256_sqrt_pd(abs_2));
  }
  Vec256<std::complex<double>> real() const {
    return Vec256<std::complex<double>>(real_values);
  }
  __m256d real_() const {return real_values;}
  Vec256<std::complex<double>> imag() const {
    return Vec256<std::complex<double>>(imag_values);
  }
  __m256d imag_() const {return imag_values;}
  Vec256<std::complex<double>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<double>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<double>> atan() const {
    return map(std::atan);
  }
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
    return map(std::exp);
  }
  Vec256<std::complex<double>> expm1() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log2() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log1p() const {
    AT_ERROR("not supported for complex numbers");
  }
  Vec256<std::complex<double>> log() const {
    return map(std::log);
  }
  Vec256<std::complex<double>> log10() const {
    return map(std::log10);
  }
  Vec256<std::complex<double>> ceil() const {
    return Vec256<std::complex<double>>(_mm256_ceil_pd(real_values), _mm256_ceil_pd(imag_values));
  }
  Vec256<std::complex<double>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<double>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<double>> floor() const {
    return Vec256<std::complex<double>>(_mm256_floor_pd(real_values), _mm256_floor_pd(imag_values));
  }
  Vec256<std::complex<double>> neg() const {
    auto zero = _mm256_setzero_pd();
    return Vec256<std::complex<double>>(_mm256_sub_pd(zero, real_values), _mm256_sub_pd(zero, imag_values));
  }
  Vec256<std::complex<double>> round() const {
    return Vec256<std::complex<double>>(_mm256_round_pd(real_values, 0), _mm256_round_pd(imag_values, 0));
  }
  Vec256<std::complex<double>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<double>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<double>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<double>> tanh() const {
    return map(std::tanh);
  }
  Vec256<std::complex<double>> trunc() const {
    return map(at::native::trunc_impl);
  }
  Vec256<std::complex<double>> sqrt() const {
    return map(std::sqrt);
  }
  Vec256<std::complex<double>> reciprocal() const {
    return map([](const std::complex<double> &x) { return (std::complex<double>)(1)/x; });
  }
  Vec256<std::complex<double>> rsqrt() const {
    return map([](const std::complex<double> &x) { return (std::complex<double>)(1)/std::sqrt(x); });
  }
  Vec256<std::complex<double>> pow(const Vec256<std::complex<double>> &exp) const {
    AT_ERROR("not supported for complex numbers");
  }
  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<std::complex<double>> operator==(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_EQ_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_EQ_OQ);
    return Vec256<std::complex<double>>(_mm256_and_pd(real_cmp, imag_cmp), _mm256_setzero_pd());
  }
  Vec256<std::complex<double>> operator!=(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_NEQ_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_NEQ_OQ);
    return Vec256<std::complex<double>>(_mm256_and_pd(real_cmp, imag_cmp), _mm256_setzero_pd());
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
  return Vec256<std::complex<double>>(_mm256_add_pd(a.real_(), b.real_()), _mm256_add_pd(a.imag_(), b.imag_()));
}

template <> Vec256<std::complex<double>> inline operator-(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  return Vec256<std::complex<double>>(_mm256_sub_pd(a.real_(), b.real_()), _mm256_sub_pd(a.imag_(), b.imag_()));
}

template <> Vec256<std::complex<double>> inline operator*(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) {
  //(a + bi)*(c + di) = (ac - bd) + (ad + bc)i.
  auto ac = _mm256_mul_pd(a.real_(), b.real_());
  auto bd = _mm256_mul_pd(a.imag_(), b.imag_());
  auto ad = _mm256_mul_pd(a.real_(), b.imag_());
  auto bc = _mm256_mul_pd(a.imag_(), b.real_());
  return Vec256<std::complex<double>>(_mm256_sub_pd(ac, bd), _mm256_add_pd(ad, bc));
}

template <> Vec256<std::complex<double>> inline operator/(const Vec256<std::complex<double>> &a, const Vec256<std::complex<double>> &b) __ubsan_ignore_float_divide_by_zero__ {
  AT_ERROR("not supported for complex numbers");
}

#endif

}}}
