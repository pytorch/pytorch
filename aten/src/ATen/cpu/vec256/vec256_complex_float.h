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

template <> class Vec256<std::complex<float>> {
private:
  __m256 real_values;
  __m256 imag_values;
public:
  static constexpr int size = 8;
  Vec256() {}
  Vec256(__m256 real) : real_values(real), imag_values(_mm256_setzero_ps()) {}
  Vec256(__m256 real, __m256 imag) : real_values(real), imag_values(imag) {}
  Vec256(std::complex<float> val) {
    real_values = _mm256_set1_ps(val.real());
    imag_values = _mm256_set1_ps(val.imag());
  }
  Vec256(std::complex<float> val1, std::complex<float> val2, std::complex<float> val3, std::complex<float> val4,
        std::complex<float> val5, std::complex<float> val6, std::complex<float> val7, std::complex<float> val8) {

    real_values = _mm256_setr_ps(val1.real(), val2.real(), val3.real(), val4.real(),
      val5.real(), val6.real(), val7.real(), val8.real());

    imag_values = _mm256_setr_ps(val1.imag(), val2.imag(), val3.imag(), val4.imag(),
      val5.imag(), val6.imag(), val7.imag(), val8.imag());
  }

  operator __m256() const {
    return real_values;
  }

  __m256 real() const {return real_values;};
  __m256 imag() const {return imag_values;};

  static Vec256<std::complex<float>> loadu(const void* ptr, int64_t count = size) {
      __at_align32__ float tmp_reals[size];
      __at_align32__ float tmp_imags[size];

      const float * tmp = reinterpret_cast<const float *>(ptr);
      for (int i=0; i<count; i++)
      {
        tmp_reals[i] = tmp[2*i];
        tmp_imags[i] = tmp[2*i+1];
      }
      return Vec256<std::complex<float>>(_mm256_loadu_ps(tmp_reals), _mm256_loadu_ps(tmp_imags));
  }

  void store(void* ptr, int count = size) const {
      float tmp[2*size];
      _mm256_storeu_ps(tmp, real_values);
      _mm256_storeu_ps(tmp + size, imag_values);

      // permute to correct order
      float *tmp_ptr = reinterpret_cast<float *>(ptr);

      for (int i=0; i<size; i++)
      {
        tmp_ptr[2*i] = tmp[i];
        tmp_ptr[2*i+1] = tmp[i+size];
      }
  }

  const std::complex<float>& operator[](int idx) const  = delete;
  std::complex<float>& operator[](int idx) = delete;

  // NOTE: this is different from real numbers, we are using math functions in <complex>, which has
  // signature: std::complex<T> (*f)(const std::complex<T> &)
  Vec256<std::complex<float>> map(std::complex<float> (*f)(const std::complex<float> &)) const {
    __at_align32__ std::complex<float> tmp[8];
    store(tmp);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = f(tmp[i]);
    }
    return loadu(tmp);
  }

  // Vec256<std::complex<float>> abs() const {
  //   if (isreal()) {
  //     auto mask = _mm256_set1_ps(-0.f);
  //     return _mm256_andnot_ps(mask, real_values);
  //   }
  //   return map(std::abs);
  // }

  // TODO: use Sleef real functions to simulate simd complex
  Vec256<std::complex<float>> acos() const {
    return map(std::acos);
  }
  Vec256<std::complex<float>> asin() const {
    return map(std::asin);
  }
  Vec256<std::complex<float>> atan() const {
    return map(std::atan);
  }
  Vec256<std::complex<float>> exp() const {
    return map(std::exp);
  }
  Vec256<std::complex<float>> log() const {
    return map(std::log);
  }
  Vec256<std::complex<float>> sin() const {
    return map(std::sin);
  }
  Vec256<std::complex<float>> sinh() const {
    return map(std::sinh);
  }
  Vec256<std::complex<float>> cos() const {
    return map(std::cos);
  }
  Vec256<std::complex<float>> cosh() const {
    return map(std::cosh);
  }
  Vec256<std::complex<float>> tan() const {
    return map(std::tan);
  }
  Vec256<std::complex<float>> tanh() const {
    return map(std::tanh);
  }
  // Vec256<float> trunc() const {
  //   return _mm256_round_ps(values, (_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
  // }
  // Vec256<float> sqrt() const {
  //   return _mm256_sqrt_ps(values);
  // }
  // Vec256<float> reciprocal() const {
  //   return _mm256_div_ps(_mm256_set1_ps(1), values);
  // }
  // Vec256<float> rsqrt() const {
  //   return _mm256_div_ps(_mm256_set1_ps(1), _mm256_sqrt_ps(values));
  // }
  // Vec256<float> pow(const Vec256<float> &b) const {
  //   return Vec256<float>(Sleef_powd4_u10(values, b));
  // }

  // Comparison using the _CMP_**_OQ predicate.
  //   `O`: get false if an operand is NaN
  //   `Q`: do not raise if an operand is NaN
  Vec256<std::complex<float>> operator==(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_EQ_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_EQ_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<float>> operator!=(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_NEQ_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_NEQ_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<float>> operator<(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_LT_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_LT_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<float>> operator<=(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_LE_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_LE_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<float>> operator>(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_GT_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_GT_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }

  Vec256<std::complex<float>> operator>=(const Vec256<std::complex<float>>& other) const {
    auto real_cmp = _mm256_cmp_ps(real_values, other.real(), _CMP_GE_OQ);
    auto imag_cmp = _mm256_cmp_ps(imag_values, other.imag_values, _CMP_GE_OQ);

    Vec256<std::complex<float>> r(real_cmp, imag_cmp);
    return r;
  }
};

template <>
Vec256<std::complex<float>> inline operator+(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  return Vec256<std::complex<float>>(
    _mm256_add_ps(a.real(), b.real()), _mm256_add_ps(a.imag(), b.imag())
  );
}

template <>
Vec256<std::complex<float>> inline operator-(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
  return Vec256<std::complex<float>>(
    _mm256_sub_ps(a.real(), b.real()), _mm256_add_ps(a.imag(), b.imag())
  );
}

// TODO: add complex mul/div for AVX2
template <>
Vec256<std::complex<float>> inline operator*(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
    __at_align32__ std::complex<float> tmp[8];
    __at_align32__ std::complex<float> tmp_a[8];
    __at_align32__ std::complex<float> tmp_b[8];

    Vec256<std::complex<float>> r;
    a.store(tmp_a);
    b.store(tmp_b);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = tmp_a[i] * tmp_b[i];
    }
    return r.loadu(tmp);
}

template <>
Vec256<std::complex<float>> inline operator/(const Vec256<std::complex<float>>& a, const Vec256<std::complex<float>>& b) {
    __at_align32__ std::complex<float> tmp[8];
    __at_align32__ std::complex<float> tmp_a[8];
    __at_align32__ std::complex<float> tmp_b[8];

    Vec256<std::complex<float>> r;
    a.store(tmp_a);
    b.store(tmp_b);
    for (int64_t i = 0; i < 8; i++) {
      tmp[i] = tmp_a[i] / tmp_b[i];
    }
    return r.loadu(tmp);
}

// #ifdef __AVX2__
// template <>
// Vec256<float> inline fmadd(const Vec256<float>& a, const Vec256<float>& b, const Vec256<float>& c) {
//   return _mm256_fmadd_ps(a, b, c);
// }
// #endif

#endif

}}}
