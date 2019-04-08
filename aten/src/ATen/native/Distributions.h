#pragma once

#include <TH/THMath.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>

// ROCM hcc doesn't work well with using std:: in kernel functions
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)
#include <c10/cuda/CUDAMathCompat.h>
#define compat_exp c10::cuda::compat::exp
#define compat_floor c10::cuda::compat::floor
#define compat_log c10::cuda::compat::log
#define compat_pow c10::cuda::compat::pow
#define compat_sqrt c10::cuda::compat::sqrt
#define compat_tan c10::cuda::compat::tan
#else
#define compat_exp std::exp
#define compat_floor std::floor
#define compat_log std::log
#define compat_pow std::pow
#define compat_sqrt std::sqrt
#define compat_tan std::tan
#endif

namespace {

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)
// we cannot use std::isnan directly due to some incompatibility of
// gcc constexpr'ing and nvcc
#define isnan std::isnan
#endif

// Here sampler_t should be function type scalar_t(void). For gpu
// "sampler" is a device function, but since ROCM doesn't have
// equivalent to nvstd::function, we use a template type parameter to
// capture it.
template<typename scalar_t, typename sampler_t>
struct BaseSampler {
  sampler_t sampler;
  C10_DEVICE BaseSampler(const sampler_t& sampler): sampler(sampler) {}
  C10_DEVICE scalar_t sample() {
    return sampler();
  }
};

// The function `sample_gamma` is
// is adapted from Numpy's distributions.c implementation.
// It is MIT licensed, so here is the copyright:

/* Copyright 2005 Robert Kern (robert.kern@gmail.com)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

template<typename scalar_t, typename accscalar_t, typename uniform_sampler_t, typename normal_sampler_t>
C10_DEVICE scalar_t sample_gamma(scalar_t alpha, BaseSampler<accscalar_t, uniform_sampler_t>& standard_uniform, BaseSampler<accscalar_t, normal_sampler_t>& standard_normal) {
  accscalar_t scale = 1.0f;

  // Boost alpha for higher acceptance probability.
  if (alpha < 1.0f) {
    if (alpha == 0.f) return 0.f;
    scale *= compat_pow(1 - standard_uniform.sample(), 1.0f / alpha);
    alpha += 1.0f;
  }

  // This implements the acceptance-rejection method of Marsaglia and Tsang (2000)
  // doi:10.1145/358407.358414
  const accscalar_t d = alpha - 1.0f / 3.0f;
  const accscalar_t c = 1.0f / compat_sqrt(9.0f * d);
  for (;;) {
    accscalar_t x, y;
    do {
      x = standard_normal.sample();
      y = 1.0f + c * x;
    } while (y <= 0);
    const accscalar_t v = y * y * y;
    const accscalar_t u = 1 - standard_uniform.sample();
    const accscalar_t xx = x * x;
    if (u < 1.0f - 0.0331f * xx * xx)
      return static_cast<scalar_t>(scale * d * v);
    if (compat_log(u) < 0.5f * xx + d * (1.0f - v + compat_log(v)))
      return static_cast<scalar_t>(scale * d * v);
  }
}

template <typename scalar_t>
C10_DEVICE static inline scalar_t polevl(const scalar_t x,  const scalar_t A[], size_t len) {
  scalar_t result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}


/*
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
template<typename scalar_t, typename accscalar_t>
C10_DEVICE static inline scalar_t digamma_one(scalar_t x) {
  constexpr accscalar_t PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }
  accscalar_t additional_summand = 0;
  int x_is_integer = x == compat_floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // it is more standard to write this as recursion, but
    // nvcc does not like that
    additional_summand = -static_cast<accscalar_t>(M_PI) /
        compat_tan(static_cast<accscalar_t>(M_PI) * x);
    x = 1 - x;
  }

  // Push x to be >= 10
  accscalar_t result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10 + additional_summand;
  }

  // Compute asymptotic digamma
  static const accscalar_t A[] = {
     8.33333333333333333333E-2,
    -2.10927960927960927961E-2,
     7.57575757575757575758E-3,
    -4.16666666666666666667E-3,
     3.96825396825396825397E-3,
    -8.33333333333333333333E-3,
     8.33333333333333333333E-2,
  };

  accscalar_t y = 0;
  if (x < 1.0e17f) {
    accscalar_t z = 1.0 / (x * x);
    y = z * polevl<accscalar_t>(z, A, 6);
  }
  return static_cast<scalar_t>(
      result + compat_log(x) - (0.5f / x) - y + additional_summand);
}

// Computes the reparameterized gradient -(d/dalpha cdf(x;alpha)) / pdf(x;alpha)
// for random number x drawn from a standard Gamma distribution Gamma(alpha).
template <typename scalar_t, typename accscalar_t>
C10_DEVICE scalar_t standard_gamma_grad_one(scalar_t alpha_, scalar_t x_) {
  // Use a Taylor series expansion for small x.
  accscalar_t x = static_cast<accscalar_t>(x_);
  accscalar_t alpha = static_cast<accscalar_t>(alpha_);
  if (x < 0.8f) {
    accscalar_t numer = 1;
    accscalar_t denom = alpha;
    auto series1 = numer / denom;
    auto series2 = numer / (denom * denom);
    for (int i = 1; i <= 5; ++i) {
      numer *= -x / static_cast<accscalar_t>(i);
      denom += 1;
      series1 += numer / denom;
      series2 += numer / (denom * denom);
    }
    const auto pow_x_alpha = compat_pow(x, alpha);
    const auto gamma_pdf = compat_pow(x, alpha - 1) * compat_exp(-x);
    const auto gamma_cdf = pow_x_alpha * series1;
    const auto gamma_cdf_alpha =
        (compat_log(x) - digamma_one<accscalar_t, accscalar_t>(alpha)) *
            gamma_cdf -
        pow_x_alpha * series2;
    const auto result = -gamma_cdf_alpha / gamma_pdf;
    return isnan(result) ? static_cast<scalar_t>( 0.f ) : static_cast<scalar_t>(result);
  }

  // Use a Rice saddle point expansion for large alpha.
  if (alpha > 8.0f) {
    if (0.9f * alpha <= x && x <= 1.1f * alpha) {
      const auto numer_1 = 1 + 24 * alpha * (1 + 12 * alpha);
      const auto numer_2 = 1440 * (alpha * alpha) + 6 * x * (53 - 120 * x)
          - 65 * x * x / alpha + alpha * (107 + 3600 * x);
      const auto denom = 1244160 * (alpha * alpha) * (alpha * alpha);
      return static_cast<scalar_t>(numer_1 * numer_2 / denom);
    }
    const auto denom = compat_sqrt(8 * alpha);
    const auto term2 = denom / (alpha - x);
    const auto term3 = compat_pow(
        x - alpha - alpha * compat_log(x / alpha),
        static_cast<accscalar_t>(-1.5));
    const auto term23 = (x < alpha) ? term2 - term3 : term2 + term3;
    const auto term1 = compat_log(x / alpha) * term23 -
        compat_sqrt(2 / alpha) * (alpha + x) / ((alpha - x) * (alpha - x));
    const auto stirling = 1 + 1 / (12 * alpha) * (1 + 1 / (24 * alpha));
    const auto numer = x * term1;
    return static_cast<scalar_t>(-stirling * numer / denom);
  }

  // Use a bivariate rational approximation to the reparameterized gradient.
  const auto u = compat_log(x / alpha);
  const auto v = compat_log(alpha);
  static const accscalar_t coef_uv[3][8] = {
    {0.16009398, -0.094634809, 0.025146376, -0.0030648343,
     1, 0.32668115, 0.10406089, 0.0014179084},
    {0.53487893, 0.1298071, 0.065735949, -0.0015649758,
     0.16639465, 0.020070113, -0.0035938915, -0.00058392623},
    {0.040121004, -0.0065914022, -0.0026286047, -0.0013441777,
     0.017050642, -0.0021309326, 0.00085092367, -1.5247877e-07},
  };
  accscalar_t coef_v[8];
  for (int i = 0; i < 8; ++ i) {
    coef_v[i] = coef_uv[0][i] + u * (coef_uv[1][i] + u * coef_uv[2][i]);
  }
  const auto p = coef_v[0] + v * (coef_v[1] + v * (coef_v[2] + v * coef_v[3]));
  const auto q = coef_v[4] + v * (coef_v[5] + v * (coef_v[6] + v * coef_v[7]));
  return static_cast<scalar_t>(compat_exp(p / q));
}

} // namespace
