#pragma once

#include "intrinsics.h"
#include "vec256_base.h"
#if defined(__AVX__) && !defined(_MSC_VER)
#include <sleef.h>
#endif

namespace at {
namespace vec256 {
namespace {

#if defined(__AVX__) && !defined(_MSC_VER)

template <> class Vec256<std::complex<double>> {
private:
  __m256d real_values;
  __m256d imag_values;
public:
  static constexpr int size = 4;
  Vec256() {}
  Vec256(__m256d real) : real_values(real), imag_values(_mm256_setzero_pd()) {}
  Vec256(__m256d real, __m256d imag) : real_values(real), imag_values(imag) {}
  Vec256(std::complex<double> val) {
    real_values = _mm256_set1_pd(val.real());
    imag_values = _mm256_set1_pd(val.imag());
  }
  Vec256(std::complex<double> val1, std::complex<double> val2, std::complex<double> val3, std::complex<double> val4) {
    real_values = _mm256_setr_pd(val1.real(), val2.real(), val3.real(), val4.real());
    imag_values = _mm256_setr_pd(val1.imag(), val2.imag(), val3.imag(), val4.imag());
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
                            int64_t count = size) {
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

  static Vec256<std::complex<double>> loadu(const void* ptr, int64_t count = size) {
      __at_align32__ double tmp_reals[size];
      __at_align32__ double tmp_imags[size];

      const double * tmp = reinterpret_cast<const double *>(ptr);
      for (int i=0; i<count; i++)
      {
        tmp_reals[i] = tmp[2*i];
        tmp_imags[i] = tmp[2*i+1];
      }
      return Vec256<std::complex<double>>(_mm256_loadu_pd(tmp_reals), _mm256_loadu_pd(tmp_imags));
  }

  void store(void* ptr, int count = size) const {
      double tmp[2*size];
      _mm256_storeu_pd(tmp, real_values);
      _mm256_storeu_pd(tmp + 4, imag_values);

      // permute to correct order
      double *tmp_ptr = reinterpret_cast<double *>(ptr);

      tmp_ptr[0] = tmp[0];
      tmp_ptr[1] = tmp[4];
      tmp_ptr[2] = tmp[1];
      tmp_ptr[3] = tmp[5];
      tmp_ptr[4] = tmp[2];
      tmp_ptr[5] = tmp[6];
      tmp_ptr[6] = tmp[3];
      tmp_ptr[7] = tmp[7];
  }

  const std::complex<double>& operator[](int idx) const  = delete;
  std::complex<double>& operator[](int idx) = delete;

  Vec256<std::complex<double>> map(std::complex<double> (*f)(const std::complex<double> &)) const {
    __at_align32__ std::complex<double> tmp[4];
    store(tmp);
    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  bool isreal() const {
    auto zeros = _mm256_setzero_pd();
    auto comparison_out = _mm256_cmp_pd(real_values, zeros, _CMP_EQ_OQ);
    return _mm256_movemask_pd(comparison_out) == 0;
  }

  bool isimag() const {
    auto zeros = _mm256_setzero_pd();
    auto comparison_out = _mm256_cmp_pd(imag_values, zeros, _CMP_EQ_OQ);
    return _mm256_movemask_pd(comparison_out) == 0;
  }

  __m256d real() const {return real_values;};
  __m256d imag() const {return imag_values;};

  // NOTE: std::abs signature is
  // template <typename T>
  // T abs(const complex<T> &);
  Vec256<double> abs() const {
    if (isreal()) {
      auto mask = _mm256_set1_pd(-0.f);
      return _mm256_andnot_pd(mask, real_values);
    }
    // FIXME: return map(std::abs);
  }

  // TODO: use Sleef real functions to simulate simd complex
  Vec256<std::complex<double>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<double>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<double>> atan() const {
    return map(std::atan);
  }
  Vec256<std::complex<double>> exp() const {
    return map(std::exp);
  }
  Vec256<std::complex<double>> log() const {
    return map(std::log);
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
  Vec256<std::complex<double>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<double>> tanh() const {
    return map(std::tanh);
  }
  // Vec256<double> trunc() const {
  //   return _mm256_round_pd(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  // }
  // Vec256<double> sqrt() const {
  //   return _mm256_sqrt_pd(values);
  // }
  // Vec256<double> reciprocal() const {
  //   return _mm256_div_pd(_mm256_set1_pd(1), values);
  // }
  // Vec256<double> rsqrt() const {
  //   return _mm256_div_pd(_mm256_set1_pd(1), _mm256_sqrt_pd(values));
  // }
  // Vec256<double> pow(const Vec256<double> &b) const {
  //   return Vec256<double>(Sleef_powd4_u10(values, b));
  // }

  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<std::complex<double>> operator==(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_EQ_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_EQ_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<double>> operator!=(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_NEQ_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_NEQ_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<double>> operator<(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_LT_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_LT_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<double>> operator<=(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_LE_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_LE_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<double>> operator>(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_GT_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_GT_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<double>> operator>=(const Vec256<std::complex<double>>& other) const {
    auto real_cmp = _mm256_cmp_pd(real_values, other.real_values, _CMP_GE_OQ);
    auto imag_cmp = _mm256_cmp_pd(imag_values, other.imag_values, _CMP_GE_OQ);

    Vec256<std::complex<double>> r(real_cmp, imag_cmp);
    return r;
  }
};

template <>
Vec256<std::complex<double>> inline operator+(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  return Vec256<std::complex<double>>(
    _mm256_add_pd(a.real(), b.real()), _mm256_add_pd(a.imag(), b.imag())
  );
}

template <>
Vec256<std::complex<double>> inline operator-(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
  return Vec256<std::complex<double>>(
    _mm256_sub_pd(a.real(), b.real()), _mm256_add_pd(a.imag(), b.imag())
  );
}

// TODO: add complex mul/div for AVX2
template <>
Vec256<std::complex<double>> inline operator*(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
    __at_align32__ std::complex<double> tmp[4];
    __at_align32__ std::complex<double> tmp_a[4];
    __at_align32__ std::complex<double> tmp_b[4];
    a.store(tmp_a);
    b.store(tmp_b);
    Vec256<std::complex<double>> r;

    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = tmp_a[i] * tmp_b[i];
    }
    return r.loadu(tmp);
}

template <>
Vec256<std::complex<double>> inline operator/(const Vec256<std::complex<double>>& a, const Vec256<std::complex<double>>& b) {
    __at_align32__ std::complex<double> tmp[4];
    __at_align32__ std::complex<double> tmp_a[4];
    __at_align32__ std::complex<double> tmp_b[4];
    a.store(tmp_a);
    b.store(tmp_b);
    Vec256<std::complex<double>> r;

    for (int64_t i = 0; i < 4; i++) {
      tmp[i] = tmp_a[i] / tmp_b[i];
    }
    return r.loadu(tmp);
}

// #ifdef __AVX2__
// template <>
// Vec256<double> inline fmadd(const Vec256<double>& a, const Vec256<double>& b, const Vec256<double>& c) {
//   return _mm256_fmadd_pd(a, b, c);
// }
// #endif

#endif

}}}
