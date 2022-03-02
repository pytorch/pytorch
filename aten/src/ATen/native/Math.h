#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <ATen/NumericUtils.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/MathConstants.h>
#include <c10/util/math_compat.h>
#include <ATen/AccumulateType.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

/* The next function is taken from  https://github.com/antelopeusersgroup/antelope_contrib/blob/master/lib/location/libgenloc/erfinv.c.
Below is the copyright.
Output was modified to be inf or -inf when input is 1 or -1. */


/*
    Copyright (c) 2014 Indiana University
    All rights reserved.

    Written by Prof. Gary L. Pavlis, Dept. of Geol. Sci.,
            Indiana University, Bloomington, IN

    This software is licensed under the New BSD license:

    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:

    Redistributions of source code must retain the above
    copyright notice, this list of conditions and the
    following disclaimer.

    Redistributions in binary form must reproduce the
    above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or
    other materials provided with the distribution.

    Neither the name of Indiana University nor
    the names of its contributors may be used to endorse
    or promote products derived from this software without
    specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
*/

#define CENTRAL_RANGE 0.7

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfinv(T y) {
/* Function to calculate inverse error function.  Rational approximation
is used to generate an initial approximation, which is then improved to
full accuracy by two steps of Newton's method.  Code is a direct
translation of the erfinv m file in matlab version 2.0.
Author:  Gary L. Pavlis, Indiana University
Date:  February 1996
*/
  T x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */
  T a[4] = {  T(0.886226899), T(-1.645349621),  T(0.914624893), T(-0.140543331) };
  T b[4] = { T(-2.118377725),  T(1.442710462), T(-0.329097515),  T(0.012229801) };
  T c[4] = { T(-1.970840454), T(-1.624906493),  T(3.429567803),  T(1.641345311) };
  T d[2] = {  T(3.543889200),  T(1.637067800) };
  T y_abs = std::abs(y);
  if(y_abs > 1.0) return std::numeric_limits<T>::quiet_NaN();
#ifdef _WIN32
  // error C2039: '_copysign': is not a member of 'std'
  if(y_abs == 1.0) return copysign(std::numeric_limits<T>::infinity(), y);
#else
  if(y_abs == 1.0) return std::copysign(std::numeric_limits<T>::infinity(), y);
#endif
  if(y_abs <= static_cast<T>(CENTRAL_RANGE)) {
    z = y * y;
    num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
    dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0]) * z + static_cast<T>(1.0));
    x = y * num / dem;
  }
  else{
    z = std::sqrt(-std::log((static_cast<T>(1.0)-y_abs)/static_cast<T>(2.0)));
    num = ((c[3]*z + c[2])*z + c[1]) * z + c[0];
    dem = (d[1]*z + d[0])*z + static_cast<T>(1.0);
#ifdef _WIN32
    // error C2039: '_copysign': is not a member of 'std'
    x = copysign(num, y) / dem;
#else
    x = std::copysign(num, y) / dem;
#endif
  }
  /* Two steps of Newton-Raphson correction */
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(c10::pi<double>)))*std::exp(-x*x));
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(c10::pi<double>)))*std::exp(-x*x));

  return(x);
}

#undef CENTRAL_RANGE

/*
 * Note [3-Clause BSD License for the Cephes Math Library]
 * Code derived from implementations in the Cephes Math Library should mention its derivation and reference
 * this note (ex. 'This function is derived from the implementation of X in the Cephes Math Library. See note
 * [3-Clause BSD License for the Cephes Math Library]. The license is:
 * Copyright (c) 2018, Steven Moshier
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of the nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Steven Moshier BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This function is derived from the implementation of the zeta function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
template <typename scalar_t, bool is_cuda=false>
C10_HOST_DEVICE static inline scalar_t zeta(scalar_t x, scalar_t q) __ubsan_ignore_float_divide_by_zero__ {
  using acc_t = at::acc_type<scalar_t, is_cuda>;
  const acc_t MACHEP = acc_t{1.11022302462515654042E-16};
  constexpr acc_t zero = acc_t{0.0};
  constexpr acc_t half = acc_t{0.5};
  constexpr acc_t one = acc_t{1.0};
  static const acc_t A[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9, /*1.307674368e12/691*/
      7.47242496e10,
      -2.950130727918164224e12, /*1.067062284288e16/3617*/
      1.1646782814350067249e14, /*5.109094217170944e18/43867*/
      -4.5979787224074726105e15, /*8.028576626982912e20/174611*/
      1.8152105401943546773e17, /*1.5511210043330985984e23/854513*/
      -7.1661652561756670113e18 /*1.6938241367317436694528e27/236364091*/
  };

  int i = 0;
  acc_t a, b, k, s, t, w;
  if (x == one) {
    return std::numeric_limits<scalar_t>::infinity();
  }

  if (x < one) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }

  if (q <= zero) {
    if (q == ::floor(q)) {
      return std::numeric_limits<scalar_t>::infinity();
    }
    if (x != ::floor(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }

  s = ::pow(q, -x);
  a = q;
  i = 0;
  b = zero;
  while ((i < 9) || (a <= acc_t{9.0})) {
    i += 1;
    a += one;
    b = ::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return static_cast<scalar_t>(s);
    }
  };

  w = a;
  s += b * w / (x - one);
  s -= half * b;
  a = one;
  k = zero;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = ::fabs(t / s);
    if (t < MACHEP) {
      return static_cast<scalar_t>(s);
    }
    k += one;
    a *= x + k;
    b /= w;
    k += one;
  }
  return static_cast<scalar_t>(s);
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 */
template <typename T>
C10_HOST_DEVICE static inline T polevl(const T x, const T A[], size_t len) {
  T result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

static inline double trigamma(double x) __ubsan_ignore_float_divide_by_zero__ {
  double sign = +1;
  double result = 0;
  if (x < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(c10::pi<double> * x);
    result -= (c10::pi<double> * c10::pi<double>) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const double ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1./6 - ixx * (1./30 - ixx * (1./42)))) / x;
  return sign * result;
}

static inline float trigamma(float x) __ubsan_ignore_float_divide_by_zero__ {
  float sign = +1;
  float result = 0;
  if (x < 0.5f) {
    sign = -1;
    const float sin_pi_x = sinf(c10::pi<float> * x);
    result -= (c10::pi<float> * c10::pi<float>) / (sin_pi_x * sin_pi_x);
    x = 1 - x;
  }
  for (int i = 0; i < 6; ++i) {
    result += 1 / (x * x);
    x += 1;
  }
  const float ixx = 1 / (x*x);
  result += (1 + 1 / (2*x) + ixx * (1.f/6 - ixx * (1.f/30 - ixx * (1.f/42)))) / x;
  return sign * result;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
static inline double calc_digamma(double x) {
  // [C++ Standard Reference: Gamma Function] https://en.cppreference.com/w/cpp/numeric/math/tgamma
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == trunc(x);
  if (x < 0) {
    if (x_is_integer) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = std::modf(x, &q);
    return calc_digamma(1 - x) - c10::pi<double> / tan(c10::pi<double> * r);
  }

  // Push x to be >= 10
  double result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10;
  }

  // Compute asymptotic digamma
  static const double A[] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  double y = 0;
  if (x < 1.0e17) {
    double z = 1.0 / (x * x);
    y = z * polevl(z, A, 6);
  }
  return result + log(x) - (0.5 / x) - y;
}

/*
 * This function is derived from the implementation of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */
static inline float calc_digamma(float x) {
  // See [C++ Standard Reference: Gamma Function]
  static float PSI_10 = 2.25175258906672110764f;
  if (x == 0) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    return std::copysign(INFINITY, -x);
  }

  bool x_is_integer = x == truncf(x);
  if (x < 0) {
    if (x_is_integer) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is a negative integer, NaN is returned
      return std::numeric_limits<float>::quiet_NaN();
    }
    // Extracts the fractional part of x as r, since tan(pi * r) is more numerically
    // accurate than tan(pi * x). While these operations are mathematically equivalent
    // since both x and r are in radians and tan() has a periodicity of pi, in practice
    // the computation of pi * x is a source of error (when |x| > 1).
    double q, r;
    r = std::modf(x, &q);
    float pi_over_tan_pi_x = (float)(c10::pi<double> / tan(c10::pi<double> * r));
    return calc_digamma(1 - x) - pi_over_tan_pi_x;
  }

  // Push x to be >= 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    return result + PSI_10;
  }

  // Compute asymptotic digamma
  static const float A[] = {
      8.33333333333333333333E-2f,
      -2.10927960927960927961E-2f,
      7.57575757575757575758E-3f,
      -4.16666666666666666667E-3f,
      3.96825396825396825397E-3f,
      -8.33333333333333333333E-3f,
      8.33333333333333333333E-2f,
  };

  float y = 0;
  if (x < 1.0e17f) {
    float z = 1 / (x * x);
    y = z * polevl(z, A, 6);
  }
  return result + logf(x) - (0.5f / x) - y;
}

template <typename scalar_t, bool is_cuda=false>
static inline C10_HOST_DEVICE scalar_t calc_polygamma(scalar_t x, int n) {
  // already blocked if n <= 1
  const auto one = scalar_t{1};
  return ((n % 2) ? one : -one) *
      ::exp(::lgamma(static_cast<scalar_t>(n) + one)) *
      zeta<scalar_t, is_cuda>(static_cast<scalar_t>(n + 1), x);
}

// regularized lower incomplete gamma
// the regularized lower, upper incomplete gamma, as well as their
// helper functions follow SciPy's implementation

/* References
 * [igam1] "The Digital Library of Mathematical Functions", dlmf.nist.gov
 * [igam2] Maddock et. al., "Incomplete Gamma Functions",
 *     https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html
 */

/*
 * This implementation of the regularized incomplete gamma functions and
 * their helper functions are derived from the implementation of SciPy's
 * gammainc, Cephes's igam and igamc, and Boost's Lanczos approximations.
 * See NOTICE for the licenses.
 */
template <typename scalar_t>
static scalar_t ratevl(scalar_t x, const scalar_t num[], int64_t M,
    const scalar_t denom[], int64_t N) {
  // evaluating rational function, i.e., the ratio of two polynomials
  // the coefficients for numerator are given by `num` while coeffs for
  // denumerator are given by `denom`

  int64_t i, dir;
  scalar_t y, num_ans, denom_ans;
  scalar_t absx = std::fabs(x);
  const scalar_t *p;

  if (absx > 1) {
    /* Evaluate as a polynomial in 1/x. */
    dir = -1;
    p = num + M;
    y = 1 / x;
  }
  else {
    dir = 1;
    p = num;
    y = x;
  }

  /* Evaluate the numerator */
  num_ans = *p;
  p += dir;
  for (i = 1; i <= M; i++) {
    num_ans = num_ans * y + *p;
    p += dir;
  }
  /* Evaluate the denominator */
  if (absx > 1) {
    p = denom + N;
  }
  else {
    p = denom;
  }

  denom_ans = *p;
  p += dir;
  for (i = 1; i <= N; i++) {
    denom_ans = denom_ans * y + *p;
    p += dir;
  }
  if (absx > 1) {
    i = N - M;
    return std::pow(x, i) * num_ans / denom_ans;
  }
  else {
    return num_ans / denom_ans;
  }
}

// SciPy's lanczos implementation is taken from Boost
/* (C) Copyright John Maddock 2006.
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. See
 * https://www.boost.org/LICENSE_1_0.txt or see NOTICE.
 */
template <typename scalar_t>
static scalar_t lanczos_sum_expg_scaled(scalar_t x) {
  // lanczos approximation
  static const scalar_t lanczos_sum_expg_scaled_num[13] = {
    0.006061842346248906525783753964555936883222,
    0.5098416655656676188125178644804694509993,
    19.51992788247617482847860966235652136208,
    449.9445569063168119446858607650988409623,
    6955.999602515376140356310115515198987526,
    75999.29304014542649875303443598909137092,
    601859.6171681098786670226533699352302507,
    3481712.15498064590882071018964774556468,
    14605578.08768506808414169982791359218571,
    43338889.32467613834773723740590533316085,
    86363131.28813859145546927288977868422342,
    103794043.1163445451906271053616070238554,
    56906521.91347156388090791033559122686859
  };
  static const scalar_t lanczos_sum_expg_scaled_denom[13] = {
    1.,
    66.,
    1925.,
    32670.,
    357423.,
    2637558.,
    13339535.,
    45995730.,
    105258076.,
    150917976.,
    120543840.,
    39916800.,
    0.
  };
  return ratevl(x, lanczos_sum_expg_scaled_num,
      sizeof(lanczos_sum_expg_scaled_num) / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
      lanczos_sum_expg_scaled_denom,
      sizeof(lanczos_sum_expg_scaled_denom) / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}

template <typename scalar_t>
static scalar_t _igam_helper_fac(scalar_t a, scalar_t x) {
  // compute x^a * exp(-a) / gamma(a)
  // corrected from (15) and (16) in [igam2] by replacing exp(x - a) with
  // exp(a - x).

  scalar_t ax, fac, res, num, numfac;
  static scalar_t MAXLOG = std::is_same<scalar_t,double>::value ?
    7.09782712893383996843E2 : 88.72283905206835;
  static scalar_t EXP1 = 2.718281828459045;
  static scalar_t lanczos_g = 6.024680040776729583740234375;

  if (std::fabs(a - x) > 0.4 * std::fabs(a)) {
    ax = a * std::log(x) - x - std::lgamma(a);
    if (ax < -MAXLOG) {
      return 0.0;
    }
    return std::exp(ax);
  }

  fac = a + lanczos_g - 0.5;
  res = std::sqrt(fac / EXP1) / lanczos_sum_expg_scaled(a);

  if ((a < 200) && (x < 200)) {
    res *= std::exp(a - x) * std::pow(x / fac, a);
  }
  else {
    num = x - a - lanczos_g + 0.5;
    numfac = num / fac;
    res *= std::exp(a * (std::log1p(numfac) - numfac) + x * (0.5 - lanczos_g) / fac);
  }
  return res;
}

template <typename scalar_t>
static scalar_t _igam_helper_series(scalar_t a, scalar_t x) {
  // Compute igam using DLMF 8.11.4. [igam1]
  static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  static int MAXITER = 2000;

  int i;
  scalar_t ans, ax, c, r;

  ax = _igam_helper_fac(a, x);
  if (ax == 0.0) {
    return 0.0;
  }

  /* power series */
  r = a;
  c = 1.0;
  ans = 1.0;

  for (i = 0; i < MAXITER; i++) {
    r += 1.0;
    c *= x / r;
    ans += c;
    if (c <= MACHEP * ans) {
      break;
    }
  }
  return (ans * ax / a);
}

template <typename scalar_t>
static scalar_t _igamc_helper_series(scalar_t a, scalar_t x) {
  // Compute igamc using DLMF 8.7.3 [igam1]. This is related to the series in
  // _igam_helper_series but extra care is taken to avoid cancellation.

  int n;
  scalar_t fac = 1;
  scalar_t sum = 0;
  scalar_t term, logx;
  static scalar_t MAXITER = 2000;
  static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;

  for (n = 1; n < MAXITER; n++) {
    fac *= -x / n;
    term = fac / (a + n);
    sum += term;
    if (std::fabs(term) <= MACHEP * std::fabs(sum)) {
        break;
    }
  }

  logx = std::log(x);
  term = -std::expm1(a * logx - std::lgamma(1+a));
  return term - std::exp(a * logx - std::lgamma(a)) * sum;
}

template <typename scalar_t>
static scalar_t _igam_helper_asymptotic_series(scalar_t a, scalar_t x, bool igam) {
  // Compute igam/igamc using DLMF 8.12.3/8.12.4 [igam1]
  static const scalar_t d[25][25] =
    {{-3.3333333333333333e-1, 8.3333333333333333e-2, -1.4814814814814815e-2,
      1.1574074074074074e-3, 3.527336860670194e-4, -1.7875514403292181e-4,
      3.9192631785224378e-5, -2.1854485106799922e-6, -1.85406221071516e-6,
      8.296711340953086e-7, -1.7665952736826079e-7, 6.7078535434014986e-9,
      1.0261809784240308e-8, -4.3820360184533532e-9, 9.1476995822367902e-10,
      -2.551419399494625e-11, -5.8307721325504251e-11, 2.4361948020667416e-11,
      -5.0276692801141756e-12, 1.1004392031956135e-13, 3.3717632624009854e-13,
      -1.3923887224181621e-13, 2.8534893807047443e-14, -5.1391118342425726e-16,
      -1.9752288294349443e-15},
    {-1.8518518518518519e-3, -3.4722222222222222e-3, 2.6455026455026455e-3,
      -9.9022633744855967e-4, 2.0576131687242798e-4, -4.0187757201646091e-7,
      -1.8098550334489978e-5, 7.6491609160811101e-6, -1.6120900894563446e-6,
      4.6471278028074343e-9, 1.378633446915721e-7, -5.752545603517705e-8,
      1.1951628599778147e-8, -1.7543241719747648e-11, -1.0091543710600413e-9,
      4.1627929918425826e-10, -8.5639070264929806e-11, 6.0672151016047586e-14,
      7.1624989648114854e-12, -2.9331866437714371e-12, 5.9966963656836887e-13,
      -2.1671786527323314e-16, -4.9783399723692616e-14, 2.0291628823713425e-14,
      -4.13125571381061e-15},
    {4.1335978835978836e-3, -2.6813271604938272e-3, 7.7160493827160494e-4,
      2.0093878600823045e-6, -1.0736653226365161e-4, 5.2923448829120125e-5,
      -1.2760635188618728e-5, 3.4235787340961381e-8, 1.3721957309062933e-6,
      -6.298992138380055e-7, 1.4280614206064242e-7, -2.0477098421990866e-10,
      -1.4092529910867521e-8, 6.228974084922022e-9, -1.3670488396617113e-9,
      9.4283561590146782e-13, 1.2872252400089318e-10, -5.5645956134363321e-11,
      1.1975935546366981e-11, -4.1689782251838635e-15, -1.0940640427884594e-12,
      4.6622399463901357e-13, -9.905105763906906e-14, 1.8931876768373515e-17,
      8.8592218725911273e-15},
    {6.4943415637860082e-4, 2.2947209362139918e-4, -4.6918949439525571e-4,
      2.6772063206283885e-4, -7.5618016718839764e-5, -2.3965051138672967e-7,
      1.1082654115347302e-5, -5.6749528269915966e-6, 1.4230900732435884e-6,
      -2.7861080291528142e-11, -1.6958404091930277e-7, 8.0994649053880824e-8,
      -1.9111168485973654e-8, 2.3928620439808118e-12, 2.0620131815488798e-9,
      -9.4604966618551322e-10, 2.1541049775774908e-10, -1.388823336813903e-14,
      -2.1894761681963939e-11, 9.7909989511716851e-12, -2.1782191880180962e-12,
      6.2088195734079014e-17, 2.126978363279737e-13, -9.3446887915174333e-14,
      2.0453671226782849e-14},
    {-8.618882909167117e-4, 7.8403922172006663e-4, -2.9907248030319018e-4,
      -1.4638452578843418e-6, 6.6414982154651222e-5, -3.9683650471794347e-5,
      1.1375726970678419e-5, 2.5074972262375328e-10, -1.6954149536558306e-6,
      8.9075075322053097e-7, -2.2929348340008049e-7, 2.956794137544049e-11,
      2.8865829742708784e-8, -1.4189739437803219e-8, 3.4463580499464897e-9,
      -2.3024517174528067e-13, -3.9409233028046405e-10, 1.8602338968504502e-10,
      -4.356323005056618e-11, 1.2786001016296231e-15, 4.6792750266579195e-12,
      -2.1492464706134829e-12, 4.9088156148096522e-13, -6.3385914848915603e-18,
      -5.0453320690800944e-14},
    {-3.3679855336635815e-4, -6.9728137583658578e-5, 2.7727532449593921e-4,
      -1.9932570516188848e-4, 6.7977804779372078e-5, 1.419062920643967e-7,
      -1.3594048189768693e-5, 8.0184702563342015e-6, -2.2914811765080952e-6,
      -3.252473551298454e-10, 3.4652846491085265e-7, -1.8447187191171343e-7,
      4.8240967037894181e-8, -1.7989466721743515e-14, -6.3061945000135234e-9,
      3.1624176287745679e-9, -7.8409242536974293e-10, 5.1926791652540407e-15,
      9.3589442423067836e-11, -4.5134262161632782e-11, 1.0799129993116827e-11,
      -3.661886712685252e-17, -1.210902069055155e-12, 5.6807435849905643e-13,
      -1.3249659916340829e-13},
    {5.3130793646399222e-4, -5.9216643735369388e-4, 2.7087820967180448e-4,
      7.9023532326603279e-7, -8.1539693675619688e-5, 5.6116827531062497e-5,
      -1.8329116582843376e-5, -3.0796134506033048e-9, 3.4651553688036091e-6,
      -2.0291327396058604e-6, 5.7887928631490037e-7, 2.338630673826657e-13,
      -8.8286007463304835e-8, 4.7435958880408128e-8, -1.2545415020710382e-8,
      8.6496488580102925e-14, 1.6846058979264063e-9, -8.5754928235775947e-10,
      2.1598224929232125e-10, -7.6132305204761539e-16, -2.6639822008536144e-11,
      1.3065700536611057e-11, -3.1799163902367977e-12, 4.7109761213674315e-18,
      3.6902800842763467e-13},
    {3.4436760689237767e-4, 5.1717909082605922e-5, -3.3493161081142236e-4,
      2.812695154763237e-4, -1.0976582244684731e-4, -1.2741009095484485e-7,
      2.7744451511563644e-5, -1.8263488805711333e-5, 5.7876949497350524e-6,
      4.9387589339362704e-10, -1.0595367014026043e-6, 6.1667143761104075e-7,
      -1.7562973359060462e-7, -1.2974473287015439e-12, 2.695423606288966e-8,
      -1.4578352908731271e-8, 3.887645959386175e-9, -3.8810022510194121e-17,
      -5.3279941738772867e-10, 2.7437977643314845e-10, -6.9957960920705679e-11,
      2.5899863874868481e-17, 8.8566890996696381e-12, -4.403168815871311e-12,
      1.0865561947091654e-12},
    {-6.5262391859530942e-4, 8.3949872067208728e-4, -4.3829709854172101e-4,
      -6.969091458420552e-7, 1.6644846642067548e-4, -1.2783517679769219e-4,
      4.6299532636913043e-5, 4.5579098679227077e-9, -1.0595271125805195e-5,
      6.7833429048651666e-6, -2.1075476666258804e-6, -1.7213731432817145e-11,
      3.7735877416110979e-7, -2.1867506700122867e-7, 6.2202288040189269e-8,
      6.5977038267330006e-16, -9.5903864974256858e-9, 5.2132144922808078e-9,
      -1.3991589583935709e-9, 5.382058999060575e-16, 1.9484714275467745e-10,
      -1.0127287556389682e-10, 2.6077347197254926e-11, -5.0904186999932993e-18,
      -3.3721464474854592e-12},
    {-5.9676129019274625e-4, -7.2048954160200106e-5, 6.7823088376673284e-4,
      -6.4014752602627585e-4, 2.7750107634328704e-4, 1.8197008380465151e-7,
      -8.4795071170685032e-5, 6.105192082501531e-5, -2.1073920183404862e-5,
      -8.8585890141255994e-10, 4.5284535953805377e-6, -2.8427815022504408e-6,
      8.7082341778646412e-7, 3.6886101871706965e-12, -1.5344695190702061e-7,
      8.862466778790695e-8, -2.5184812301826817e-8, -1.0225912098215092e-14,
      3.8969470758154777e-9, -2.1267304792235635e-9, 5.7370135528051385e-10,
      -1.887749850169741e-19, -8.0931538694657866e-11, 4.2382723283449199e-11,
      -1.1002224534207726e-11},
    {1.3324454494800656e-3, -1.9144384985654775e-3, 1.1089369134596637e-3,
      9.932404122642299e-7, -5.0874501293093199e-4, 4.2735056665392884e-4,
      -1.6858853767910799e-4, -8.1301893922784998e-9, 4.5284402370562147e-5,
      -3.127053674781734e-5, 1.044986828530338e-5, 4.8435226265680926e-11,
      -2.1482565873456258e-6, 1.329369701097492e-6, -4.0295693092101029e-7,
      -1.7567877666323291e-13, 7.0145043163668257e-8, -4.040787734999483e-8,
      1.1474026743371963e-8, 3.9642746853563325e-18, -1.7804938269892714e-9,
      9.7480262548731646e-10, -2.6405338676507616e-10, 5.794875163403742e-18,
      3.7647749553543836e-11},
    {1.579727660730835e-3, 1.6251626278391582e-4, -2.0633421035543276e-3,
      2.1389686185689098e-3, -1.0108559391263003e-3, -3.9912705529919201e-7,
      3.6235025084764691e-4, -2.8143901463712154e-4, 1.0449513336495887e-4,
      2.1211418491830297e-9, -2.5779417251947842e-5, 1.7281818956040463e-5,
      -5.6413773872904282e-6, -1.1024320105776174e-11, 1.1223224418895175e-6,
      -6.8693396379526735e-7, 2.0653236975414887e-7, 4.6714772409838506e-14,
      -3.5609886164949055e-8, 2.0470855345905963e-8, -5.8091738633283358e-9,
      -1.332821287582869e-16, 9.0354604391335133e-10, -4.9598782517330834e-10,
      1.3481607129399749e-10},
    {-4.0725121195140166e-3, 6.4033628338080698e-3, -4.0410161081676618e-3,
      -2.183732802866233e-6, 2.1740441801254639e-3, -1.9700440518418892e-3,
      8.3595469747962458e-4, 1.9445447567109655e-8, -2.5779387120421696e-4,
      1.9009987368139304e-4, -6.7696499937438965e-5, -1.4440629666426572e-10,
      1.5712512518742269e-5, -1.0304008744776893e-5, 3.304517767401387e-6,
      7.9829760242325709e-13, -6.4097794149313004e-7, 3.8894624761300056e-7,
      -1.1618347644948869e-7, -2.816808630596451e-15, 1.9878012911297093e-8,
      -1.1407719956357511e-8, 3.2355857064185555e-9, 4.1759468293455945e-20,
      -5.0423112718105824e-10},
    {-5.9475779383993003e-3, -5.4016476789260452e-4, 8.7910413550767898e-3,
      -9.8576315587856125e-3, 5.0134695031021538e-3, 1.2807521786221875e-6,
      -2.0626019342754683e-3, 1.7109128573523058e-3, -6.7695312714133799e-4,
      -6.9011545676562133e-9, 1.8855128143995902e-4, -1.3395215663491969e-4,
      4.6263183033528039e-5, 4.0034230613321351e-11, -1.0255652921494033e-5,
      6.612086372797651e-6, -2.0913022027253008e-6, -2.0951775649603837e-13,
      3.9756029041993247e-7, -2.3956211978815887e-7, 7.1182883382145864e-8,
      8.925574873053455e-16, -1.2101547235064676e-8, 6.9350618248334386e-9,
      -1.9661464453856102e-9},
    {1.7402027787522711e-2, -2.9527880945699121e-2, 2.0045875571402799e-2,
      7.0289515966903407e-6, -1.2375421071343148e-2, 1.1976293444235254e-2,
      -5.4156038466518525e-3, -6.3290893396418616e-8, 1.8855118129005065e-3,
      -1.473473274825001e-3, 5.5515810097708387e-4, 5.2406834412550662e-10,
      -1.4357913535784836e-4, 9.9181293224943297e-5, -3.3460834749478311e-5,
      -3.5755837291098993e-12, 7.1560851960630076e-6, -4.5516802628155526e-6,
      1.4236576649271475e-6, 1.8803149082089664e-14, -2.6623403898929211e-7,
      1.5950642189595716e-7, -4.7187514673841102e-8, -6.5107872958755177e-17,
      7.9795091026746235e-9},
    {3.0249124160905891e-2, 2.4817436002649977e-3, -4.9939134373457022e-2,
      5.9915643009307869e-2, -3.2483207601623391e-2, -5.7212968652103441e-6,
      1.5085251778569354e-2, -1.3261324005088445e-2, 5.5515262632426148e-3,
      3.0263182257030016e-8, -1.7229548406756723e-3, 1.2893570099929637e-3,
      -4.6845138348319876e-4, -1.830259937893045e-10, 1.1449739014822654e-4,
      -7.7378565221244477e-5, 2.5625836246985201e-5, 1.0766165333192814e-12,
      -5.3246809282422621e-6, 3.349634863064464e-6, -1.0381253128684018e-6,
      -5.608909920621128e-15, 1.9150821930676591e-7, -1.1418365800203486e-7,
      3.3654425209171788e-8},
    {-9.9051020880159045e-2, 1.7954011706123486e-1, -1.2989606383463778e-1,
      -3.1478872752284357e-5, 9.0510635276848131e-2, -9.2828824411184397e-2,
      4.4412112839877808e-2, 2.7779236316835888e-7, -1.7229543805449697e-2,
      1.4182925050891573e-2, -5.6214161633747336e-3, -2.39598509186381e-9,
      1.6029634366079908e-3, -1.1606784674435773e-3, 4.1001337768153873e-4,
      1.8365800754090661e-11, -9.5844256563655903e-5, 6.3643062337764708e-5,
      -2.076250624489065e-5, -1.1806020912804483e-13, 4.2131808239120649e-6,
      -2.6262241337012467e-6, 8.0770620494930662e-7, 6.0125912123632725e-16,
      -1.4729737374018841e-7},
    {-1.9994542198219728e-1, -1.5056113040026424e-2, 3.6470239469348489e-1,
      -4.6435192311733545e-1, 2.6640934719197893e-1, 3.4038266027147191e-5,
      -1.3784338709329624e-1, 1.276467178337056e-1, -5.6213828755200985e-2,
      -1.753150885483011e-7, 1.9235592956768113e-2, -1.5088821281095315e-2,
      5.7401854451350123e-3, 1.0622382710310225e-9, -1.5335082692563998e-3,
      1.0819320643228214e-3, -3.7372510193945659e-4, -6.6170909729031985e-12,
      8.4263617380909628e-5, -5.5150706827483479e-5, 1.7769536448348069e-5,
      3.8827923210205533e-14, -3.53513697488768e-6, 2.1865832130045269e-6,
      -6.6812849447625594e-7},
    {7.2438608504029431e-1, -1.3918010932653375, 1.0654143352413968,
      1.876173868950258e-4, -8.2705501176152696e-1, 8.9352433347828414e-1,
      -4.4971003995291339e-1, -1.6107401567546652e-6, 1.9235590165271091e-1,
      -1.6597702160042609e-1, 6.8882222681814333e-2, 1.3910091724608687e-8,
      -2.146911561508663e-2, 1.6228980898865892e-2, -5.9796016172584256e-3,
      -1.1287469112826745e-10, 1.5167451119784857e-3, -1.0478634293553899e-3,
      3.5539072889126421e-4, 8.1704322111801517e-13, -7.7773013442452395e-5,
      5.0291413897007722e-5, -1.6035083867000518e-5, 1.2469354315487605e-14,
      3.1369106244517615e-6},
    {1.6668949727276811, 1.165462765994632e-1, -3.3288393225018906,
      4.4692325482864037, -2.6977693045875807, -2.600667859891061e-4,
      1.5389017615694539, -1.4937962361134612, 6.8881964633233148e-1,
      1.3077482004552385e-6, -2.5762963325596288e-1, 2.1097676102125449e-1,
      -8.3714408359219882e-2, -7.7920428881354753e-9, 2.4267923064833599e-2,
      -1.7813678334552311e-2, 6.3970330388900056e-3, 4.9430807090480523e-11,
      -1.5554602758465635e-3, 1.0561196919903214e-3, -3.5277184460472902e-4,
      9.3002334645022459e-14, 7.5285855026557172e-5, -4.8186515569156351e-5,
      1.5227271505597605e-5},
    {-6.6188298861372935, 1.3397985455142589e+1, -1.0789350606845146e+1,
      -1.4352254537875018e-3, 9.2333694596189809, -1.0456552819547769e+1,
      5.5105526029033471, 1.2024439690716742e-5, -2.5762961164755816,
      2.3207442745387179, -1.0045728797216284, -1.0207833290021914e-7,
      3.3975092171169466e-1, -2.6720517450757468e-1, 1.0235252851562706e-1,
      8.4329730484871625e-10, -2.7998284958442595e-2, 2.0066274144976813e-2,
      -7.0554368915086242e-3, 1.9402238183698188e-12, 1.6562888105449611e-3,
      -1.1082898580743683e-3, 3.654545161310169e-4, -5.1290032026971794e-11,
      -7.6340103696869031e-5},
    {-1.7112706061976095e+1, -1.1208044642899116, 3.7131966511885444e+1,
      -5.2298271025348962e+1, 3.3058589696624618e+1, 2.4791298976200222e-3,
      -2.061089403411526e+1, 2.088672775145582e+1, -1.0045703956517752e+1,
      -1.2238783449063012e-5, 4.0770134274221141, -3.473667358470195,
      1.4329352617312006, 7.1359914411879712e-8, -4.4797257159115612e-1,
      3.4112666080644461e-1, -1.2699786326594923e-1, -2.8953677269081528e-10,
      3.3125776278259863e-2, -2.3274087021036101e-2, 8.0399993503648882e-3,
      -1.177805216235265e-9, -1.8321624891071668e-3, 1.2108282933588665e-3,
      -3.9479941246822517e-4},
    {7.389033153567425e+1, -1.5680141270402273e+2, 1.322177542759164e+2,
      1.3692876877324546e-2, -1.2366496885920151e+2, 1.4620689391062729e+2,
      -8.0365587724865346e+1, -1.1259851148881298e-4, 4.0770132196179938e+1,
      -3.8210340013273034e+1, 1.719522294277362e+1, 9.3519707955168356e-7,
      -6.2716159907747034, 5.1168999071852637, -2.0319658112299095,
      -4.9507215582761543e-9, 5.9626397294332597e-1, -4.4220765337238094e-1,
      1.6079998700166273e-1, -2.4733786203223402e-8, -4.0307574759979762e-2,
      2.7849050747097869e-2, -9.4751858992054221e-3, 6.419922235909132e-6,
      2.1250180774699461e-3},
    {2.1216837098382522e+2, 1.3107863022633868e+1, -4.9698285932871748e+2,
      7.3121595266969204e+2, -4.8213821720890847e+2, -2.8817248692894889e-2,
      3.2616720302947102e+2, -3.4389340280087117e+2, 1.7195193870816232e+2,
      1.4038077378096158e-4, -7.52594195897599e+1, 6.651969984520934e+1,
      -2.8447519748152462e+1, -7.613702615875391e-7, 9.5402237105304373,
      -7.5175301113311376, 2.8943997568871961, -4.6612194999538201e-7,
      -8.0615149598794088e-1, 5.8483006570631029e-1, -2.0845408972964956e-1,
      1.4765818959305817e-4, 5.1000433863753019e-2, -3.3066252141883665e-2,
      1.5109265210467774e-2},
    {-9.8959643098322368e+2, 2.1925555360905233e+3, -1.9283586782723356e+3,
      -1.5925738122215253e-1, 1.9569985945919857e+3, -2.4072514765081556e+3,
      1.3756149959336496e+3, 1.2920735237496668e-3, -7.525941715948055e+2,
      7.3171668742208716e+2, -3.4137023466220065e+2, -9.9857390260608043e-6,
      1.3356313181291573e+2, -1.1276295161252794e+2, 4.6310396098204458e+1,
      -7.9237387133614756e-6, -1.4510726927018646e+1, 1.1111771248100563e+1,
      -4.1690817945270892, 3.1008219800117808e-3, 1.1220095449981468,
      -7.6052379926149916e-1, 3.6262236505085254e-1, 2.216867741940747e-1,
      4.8683443692930507e-1}};

  int k, n, sgn;
  int maxpow = 0;
  static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  scalar_t lambda = x / a;
  scalar_t sigma = (x - a) / a;
  scalar_t eta, res, ck, ckterm, term, absterm;
  scalar_t absoldterm = INFINITY;
  scalar_t etapow[25] = {1};
  scalar_t sum = 0;
  scalar_t afac = 1;

  if (igam) {
    sgn = -1;
  }
  else {
    sgn = 1;
  }

  if (lambda > 1) {
    eta = std::sqrt(-2 * (std::log1p(sigma) - sigma));
  }
  else if (lambda < 1) {
    eta = -std::sqrt(-2 * (std::log1p(sigma) - sigma));
  }
  else {
    eta = 0;
  }
  res = 0.5 * std::erfc(sgn * eta * std::sqrt(a / 2));

  for (k = 0; k < 25; k++) {
    ck = d[k][0];
    for (n = 1; n < 25; n++) {
      if (n > maxpow) {
        etapow[n] = eta * etapow[n-1];
        maxpow += 1;
      }
      ckterm = d[k][n]*etapow[n];
      ck += ckterm;
      if (std::fabs(ckterm) < MACHEP * std::fabs(ck)) {
        break;
      }
    }
    term = ck * afac;
    absterm = std::fabs(term);
    if (absterm > absoldterm) {
      break;
    }
    sum += term;
    if (absterm < MACHEP * std::fabs(sum)) {
      break;
    }
    absoldterm = absterm;
    afac /= a;
  }
  res += sgn * std::exp(-0.5 * a * eta * eta) * sum / std::sqrt(2 * c10::pi<float> * a);

  return res;
}

template <typename scalar_t>
static scalar_t _igamc_helper_continued_fraction(scalar_t a, scalar_t x) {
  // Compute igamc using DLMF 8.9.2. [igam1]
  int i;
  scalar_t ans, ax, c, yc, r, t, y, z;
  scalar_t pk, pkm1, pkm2, qk, qkm1, qkm2;
  int MAXITER = 2000;
  static scalar_t MACHEP = std::is_same<scalar_t, double>::value ?
    1.11022302462515654042E-16 : 5.9604644775390625E-8;
  static scalar_t BIG = std::is_same<scalar_t,double>::value ?
    4.503599627370496e15 : 16777216.;
  static scalar_t BIGINV = std::is_same<scalar_t,double>::value ?
    2.22044604925031308085e-16 : 5.9604644775390625E-8;

  ax = _igam_helper_fac(a, x);
  if (ax == 0.0) {
    return 0.0;
  }

  /* continued fraction */
  y = 1.0 - a;
  z = x + y + 1.0;
  c = 0.0;
  pkm2 = 1.0;
  qkm2 = x;
  pkm1 = x + 1.0;
  qkm1 = z * x;
  ans = pkm1 / qkm1;

  for (i = 0; i < MAXITER; i++) {
    c += 1.0;
    y += 1.0;
    z += 2.0;
    yc = y * c;
    pk = pkm1 * z - pkm2 * yc;
    qk = qkm1 * z - qkm2 * yc;
    if (qk != 0) {
      r = pk / qk;
      t = std::fabs((ans - r) / r);
      ans = r;
    }
    else {
      t = 1.0;
    }
    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;
    if (std::fabs(pk) > BIG) {
      pkm2 *= BIGINV;
      pkm1 *= BIGINV;
      qkm2 *= BIGINV;
      qkm1 *= BIGINV;
    }
    if (t <= MACHEP) {
      break;
    }
  }
  return ans * ax;
}

template <typename scalar_t>
static inline scalar_t calc_igammac(scalar_t a, scalar_t x) {
  /* the calculation of the regularized upper incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.4 [igam1])
   * - if x > 1.1 and x < a, using the substraction from the regularized lower
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (5)
   */
  scalar_t absxma_a;

  static scalar_t SMALL = 20.0;
  static scalar_t LARGE = 200.0;
  static scalar_t SMALLRATIO = 0.3;
  static scalar_t LARGERATIO = 4.5;

  // note that in SciPy, a and x are non-negative, with exclusive 0s (i.e.,
  // at most 1 of them can be 0), where igammac(0, x) = 0.0 iff x > 0.
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 0.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 1.0;
  }
  else if (std::isinf(a)) {
    if (std::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 1.0;
  }
  else if (std::isinf(x)) {
    return 0.0;
  }

  absxma_a = std::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / std::sqrt(a))) {
     return _igam_helper_asymptotic_series(a, x, 0);
  }

  if (x > 1.1) {
    if (x < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_continued_fraction(a, x);
    }
  }
  else if (x <= 0.5) {
    if (-0.4 / std::log(x) < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
  else {
    if (x * 1.1 < a) {
      return 1.0 - _igam_helper_series(a, x);
    }
    else {
      return _igamc_helper_series(a, x);
    }
  }
}

template <typename scalar_t>
static inline scalar_t calc_igamma(scalar_t a, scalar_t x) {
  /* the calculation of the regularized lower incomplete gamma function
   * is done differently based on the values of a and x:
   * - if x and/or a is at the boundary of defined region, then assign the
   *   result at the boundary
   * - if a is large and a ~ x, then using Uniform Asymptotic Expansions for
   *   Large Parameter (see DLMF 8.12.3 [igam1])
   * - if x > 1 and x > a, using the substraction from the regularized upper
   *   incomplete gamma
   * - otherwise, calculate the series from [igam2] eq (4)
   */
  scalar_t absxma_a;
  static scalar_t SMALL = 20.0;
  static scalar_t LARGE = 200.0;
  static scalar_t SMALLRATIO = 0.3;
  static scalar_t LARGERATIO = 4.5;

  // boundary values following SciPy
  // note that in SciPy, a and x are non-negative, with exclusive 0s (i.e.,
  // at most 1 of them can be 0), where igamma(0, x) = 1.0 iff x > 0.
  if ((x < 0) || (a < 0)) {
    // out of defined-region of the function
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  else if (a == 0) {
    if (x > 0) {
      return 1.0;
    }
    else {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
  }
  else if (x == 0) {
    return 0.0; // zero integration limit
  }
  else if (std::isinf(a)) {
    if (std::isinf(x)) {
      return std::numeric_limits<scalar_t>::quiet_NaN();
    }
    return 0.0;
  }
  else if (std::isinf(x)) {
    return 1.0;
  }

  /* Asymptotic regime where a ~ x. See [igam2] */
  absxma_a = std::fabs(x - a) / a;
  if ((a > SMALL) && (a < LARGE) && (absxma_a < SMALLRATIO)) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }
  else if ((a > LARGE) && (absxma_a < LARGERATIO / std::sqrt(a))) {
    return _igam_helper_asymptotic_series(a, x, 1);
  }

  if ((x > 1.0) && (x > a)) {
    return 1.0 - calc_igammac(a, x);
  }

  return _igam_helper_series(a, x);
}

template <>
C10_UNUSED c10::BFloat16 calc_igamma<c10::BFloat16>(c10::BFloat16 a, c10::BFloat16 x) {
  return calc_igamma<float>(float(a), float(x));
}

template <>
C10_UNUSED c10::Half calc_igamma<c10::Half>(c10::Half a, c10::Half x) {
  return calc_igamma<float>(float(a), float(x));
}

template <>
C10_UNUSED c10::BFloat16 calc_igammac<c10::BFloat16>(c10::BFloat16 a, c10::BFloat16 x) {
  return calc_igammac<float>(float(a), float(x));
}

template <>
C10_UNUSED c10::Half calc_igammac<c10::Half>(c10::Half a, c10::Half x) {
  return calc_igammac<float>(float(a), float(x));
}

inline c10::BFloat16 calc_erfinv(c10::BFloat16 a) { return calc_erfinv(float(a)); }

template <typename T>
static T abs_impl(T v) {
  return std::abs(v);
}

template <>
C10_UNUSED uint8_t abs_impl(uint8_t v) {
  return v;
}

template <typename T>
static inline typename std::enable_if<std::is_integral<T>::value, T>::type
calc_gcd(T a, T b) {
  a = abs_impl(a);
  b = abs_impl(b);
  while (a != 0) {
    T c = a;
    a = b % a;
    b = c;
  }
  return b;
}

/*
 * This function is derived from the implementation of the chbevl function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Evaluates the series
 *
 *       len-1
 *         - '
 *  y  =   >   array[i] T (x/2)
 *         -             i
 *        i=0
 *
 * of Chebyshev polynomials Ti at argument x/2.
 *
 * Coefficients are stored in reverse order, i.e. the zero order term is last in the array.  Note len is the number of
 * coefficients, not the order.
 *
 * If coefficients are for the interval a to b, x must have been transformed to x -> 2(2x - b - a)/(b-a) before
 * entering the routine.  This maps x from (a, b) to (-1, 1), over which the Chebyshev polynomials are defined.
 *
 * If the coefficients are for the inverted interval, in which (a, b) is mapped to (1/b, 1/a), the transformation
 * required is x -> 2(2ab/x - b - a)/(b-a).  If b is infinity, this becomes x -> 4a/x - 1.
 */
template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
chbevl(const T x, const T array[], size_t len) {
  T b0, b1, b2;

  b0 = array[0];
  b1 = static_cast<T>(0.0);

  for (size_t i = 1; i < len; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + array[i];
  }

  return (static_cast<T>(0.5) * (b0 - b2));
}

/*
 * This function is derived from the implementation of the i0 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the zeroth order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
static inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_A() {
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static const T coeff[] = {
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};
  return std::make_tuple(coeff, 30);
};

template <typename T>
static inline std::tuple<const T*, size_t> chebyshev_coefficients_i0e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  return std::make_tuple(coeff, 25);
};

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};
  return std::make_tuple(coeff, 29);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_A() {
  /* Chebyshev coefficients for exp(-x) I1(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I1(x) / x } = 1/2.
   */
  static const T coeff[] = {
      9.38153738649577178388E-9f,
      -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f,
      -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f,
      -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f,
      -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f,
      -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f,
      -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f,
      -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f,
      -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};
  return std::make_tuple(coeff, 17);
};

template <typename T>
static inline typename std::enable_if<std::is_same<double, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  return std::make_tuple(coeff, 25);
};

template <typename T>
static inline typename std::enable_if<std::is_same<float, T>::value, std::tuple<const T*, size_t>>::type
chebyshev_coefficients_i1e_B() {
  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static const T coeff[] = {
      -3.83538038596423702205E-9f,
      -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f,
      -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f,
      -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};

  return std::make_tuple(coeff, 7);
};

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i0(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    return static_cast<T>(std::exp(x) * chbevl(y, A, len));
  }
  auto coeff_pair = chebyshev_coefficients_i0e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  return std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
}

// Upcast bfloat16 input to float for numerical accuracy purposes
static inline c10::BFloat16 calc_i0(c10::BFloat16 a) { return calc_i0(static_cast<float>(a)); }

/*
 * This function is derived from the implementation of the i0e function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the exponentially scaled zeroth order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i0e(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i0e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    return chbevl(y, A, len);
  }

  auto coeff_pair = chebyshev_coefficients_i0e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  return chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
}

// Upcast bfloat16 input to float for numerical accuracy purposes
static inline c10::BFloat16 calc_i0e(c10::BFloat16 a) { return calc_i0e(static_cast<float>(a)); }

/*
 * This function is derived from the implementation of the i1 function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    const T out = std::exp(x) * x * chbevl(y, A, len);
    return (_x < T{0.0}) ? -out : out;
  }
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const T out = (std::exp(x) * chbevl(T{32.0} / x - T{2.0}, B, len)) / std::sqrt(x);
  return (_x < T{0.0}) ? -out : out;
}

/*
 * This function is derived from the implementation of the i1e function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes an approximation of the exponentially scaled first order modified Bessel function of the first kind.
 * The approximation is actually two (sub)approximations, both using a Chebyshev polynomial expansion.
 * One approximates the function over [0, 8], and the other over (8, infinity). This function takes the absolute value
 * of all inputs to convert them into the domain of the approximation.
 */
template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i1e(T _x) {
  T x = std::abs(_x);

  if (x <= T{8.0}) {
    auto coeff_pair = chebyshev_coefficients_i1e_A<T>();
    auto A = std::get<0>(coeff_pair);
    auto len = std::get<1>(coeff_pair);
    T y = (x / T{2.0}) - T{2.0};
    const T out = chbevl(y, A, len) * x;
    return (_x < T{0.0}) ? -out : out;
  }
  auto coeff_pair = chebyshev_coefficients_i1e_B<T>();
  auto B = std::get<0>(coeff_pair);
  auto len = std::get<1>(coeff_pair);
  const auto out = chbevl(T{32.0} / x - T{2.0}, B, len) / std::sqrt(x);
  return (_x < T{0.0}) ? -out : out;
}

/*
 * This function is derived from the implementation of the i1e function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 *
 * Computes the argument, x, for which the area under the Gaussian probability density function
 * (integrated from minus infinity to x) is equal to y.
 */
template <typename T>
static inline C10_HOST_DEVICE T calc_ndtri(T y0) {

  /* sqrt(2pi) */
  constexpr T s2pi = 2.50662827463100050242E0;
  constexpr T one = 1;
  constexpr T zero = 0;

  /* approximation for 0 <= |y - 0.5| <= 3/8 */
  static const T P0[5] = {
      -5.99633501014107895267E1,
      9.80010754185999661536E1,
      -5.66762857469070293439E1,
      1.39312609387279679503E1,
      -1.23916583867381258016E0,
  };

  static const T Q0[9] = {
      1.00000000000000000000E0,
      1.95448858338141759834E0,
      4.67627912898881538453E0,
      8.63602421390890590575E1,
      -2.25462687854119370527E2,
      2.00260212380060660359E2,
      -8.20372256168333339912E1,
      1.59056225126211695515E1,
      -1.18331621121330003142E0,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
  * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
  */
  static const T P1[9] = {
      4.05544892305962419923E0,
      3.15251094599893866154E1,
      5.71628192246421288162E1,
      4.40805073893200834700E1,
      1.46849561928858024014E1,
      2.18663306850790267539E0,
      -1.40256079171354495875E-1,
      -3.50424626827848203418E-2,
      -8.57456785154685413611E-4,
  };

  static const T Q1[9] = {
      1.00000000000000000000E0,
      1.57799883256466749731E1,
      4.53907635128879210584E1,
      4.13172038254672030440E1,
      1.50425385692907503408E1,
      2.50464946208309415979E0,
      -1.42182922854787788574E-1,
      -3.80806407691578277194E-2,
      -9.33259480895457427372E-4,
  };

  /* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
  * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
  */

  static const T P2[9] = {
      3.23774891776946035970E0,
      6.91522889068984211695E0,
      3.93881025292474443415E0,
      1.33303460815807542389E0,
      2.01485389549179081538E-1,
      1.23716634817820021358E-2,
      3.01581553508235416007E-4,
      2.65806974686737550832E-6,
      6.23974539184983293730E-9,
  };

  static const T Q2[9] = {
      1.00000000000000000000E0,
      6.02427039364742014255E0,
      3.67983563856160859403E0,
      1.37702099489081330271E0,
      2.16236993594496635890E-1,
      1.34204006088543189037E-2,
      3.28014464682127739104E-4,
      2.89247864745380683936E-6,
      6.79019408009981274425E-9,
  };

  if (y0 == zero) {
    return -std::numeric_limits<T>::infinity();
  }
  if (y0 == one) {
    return std::numeric_limits<T>::infinity();
  }
  if (y0 < zero || y0 > one) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  bool code = true;
  T y = y0;
  if (y > one - T{0.13533528323661269189}) { /* 0.135... = exp(-2) */
    y = one - y;
    code = false;
  }

  if (y > T{0.13533528323661269189}) {
    y = y - T{0.5};
    const T y2 = y * y;
    T x = y + y * (y2 * polevl(y2, P0, 4) / polevl(y2, Q0, 8));
    return (x * s2pi);
  }

  T x = ::sqrt(T{-2.0} * ::log(y));
  const T x0 = x - ::log(x) / x;

  const T z = one / x;
  T x1;
  if (x < T{8.0}) /* y > exp(-32) = 1.2664165549e-14 */
  {
    x1 = z * polevl(z, P1, 8) / polevl(z, Q1, 8);
  } else {
    x1 = z * polevl(z, P2, 8) / polevl(z, Q2, 8);
  }
  x = x0 - x1;
  if (code) {
    x = -x;
  }
  return x;
}

/* The next function is taken from http://ab-initio.mit.edu/Faddeev */

/* Copyright (c) 2012 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* erfcx(x) = exp(x^2) erfc(x) function, for real x, written by
   Steven G. Johnson, October 2012.

   This function combines a few different ideas.

   First, for x > 50, it uses a continued-fraction expansion (same as
   for the Faddeeva function, but with algebraic simplifications for z=i*x).

   Second, for 0 <= x <= 50, it uses Chebyshev polynomial approximations,
   but with two twists:

      a) It maps x to y = 4 / (4+x) in [0,1].  This simple transformation,
         inspired by a similar transformation in the octave-forge/specfun
         erfcx by Soren Hauberg, results in much faster Chebyshev convergence
         than other simple transformations I have examined.

      b) Instead of using a single Chebyshev polynomial for the entire
         [0,1] y interval, we break the interval up into 100 equal
         subintervals, with a switch/lookup table, and use much lower
         degree Chebyshev polynomials in each subinterval. This greatly
         improves performance in my tests.

   For x < 0, we use the relationship erfcx(-x) = 2 exp(x^2) - erfc(x),
   with the usual checks for overflow etcetera.

   Performance-wise, it seems to be substantially faster than either
   the SLATEC DERFC function [or an erfcx function derived therefrom]
   or Cody's CALERF function (from netlib.org/specfun), while
   retaining near machine precision in accuracy.  */

/* Given y100=100*y, where y = 4/(4+x) for x >= 0, compute erfc(x).

   Uses a look-up table of 100 different Chebyshev polynomials
   for y intervals [0,0.01], [0.01,0.02], ...., [0.99,1], generated
   with the help of Maple and a little shell script.   This allows
   the Chebyshev polynomials to be of significantly lower degree (about 1/4)
   compared to fitting the whole [0,1] interval with a single polynomial. */


template <typename T>
C10_HOST_DEVICE  static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
erfcx_y100(T y100)
{
  switch (static_cast<int>(y100)) {
case 0: {
T t = 2*y100 - 1;
return 0.70878032454106438663e-3 + (0.71234091047026302958e-3 + (0.35779077297597742384e-5 + (0.17403143962587937815e-7 + (0.81710660047307788845e-10 + (0.36885022360434957634e-12 + 0.15917038551111111111e-14 * t) * t) * t) * t) * t) * t;
}
case 1: {
T t = 2*y100 - 3;
return 0.21479143208285144230e-2 + (0.72686402367379996033e-3 + (0.36843175430938995552e-5 + (0.18071841272149201685e-7 + (0.85496449296040325555e-10 + (0.38852037518534291510e-12 + 0.16868473576888888889e-14 * t) * t) * t) * t) * t) * t;
}
case 2: {
T t = 2*y100 - 5;
return 0.36165255935630175090e-2 + (0.74182092323555510862e-3 + (0.37948319957528242260e-5 + (0.18771627021793087350e-7 + (0.89484715122415089123e-10 + (0.40935858517772440862e-12 + 0.17872061464888888889e-14 * t) * t) * t) * t) * t) * t;
}
case 3: {
T t = 2*y100 - 7;
return 0.51154983860031979264e-2 + (0.75722840734791660540e-3 + (0.39096425726735703941e-5 + (0.19504168704300468210e-7 + (0.93687503063178993915e-10 + (0.43143925959079664747e-12 + 0.18939926435555555556e-14 * t) * t) * t) * t) * t) * t;
}
case 4: {
T t = 2*y100 - 9;
return 0.66457513172673049824e-2 + (0.77310406054447454920e-3 + (0.40289510589399439385e-5 + (0.20271233238288381092e-7 + (0.98117631321709100264e-10 + (0.45484207406017752971e-12 + 0.20076352213333333333e-14 * t) * t) * t) * t) * t) * t;
}
case 5: {
T t = 2*y100 - 11;
return 0.82082389970241207883e-2 + (0.78946629611881710721e-3 + (0.41529701552622656574e-5 + (0.21074693344544655714e-7 + (0.10278874108587317989e-9 + (0.47965201390613339638e-12 + 0.21285907413333333333e-14 * t) * t) * t) * t) * t) * t;
}
case 6: {
T t = 2*y100 - 13;
return 0.98039537275352193165e-2 + (0.80633440108342840956e-3 + (0.42819241329736982942e-5 + (0.21916534346907168612e-7 + (0.10771535136565470914e-9 + (0.50595972623692822410e-12 + 0.22573462684444444444e-14 * t) * t) * t) * t) * t) * t;
}
case 7: {
T t = 2*y100 - 15;
return 0.11433927298290302370e-1 + (0.82372858383196561209e-3 + (0.44160495311765438816e-5 + (0.22798861426211986056e-7 + (0.11291291745879239736e-9 + (0.53386189365816880454e-12 + 0.23944209546666666667e-14 * t) * t) * t) * t) * t) * t;
}
case 8: {
T t = 2*y100 - 17;
return 0.13099232878814653979e-1 + (0.84167002467906968214e-3 + (0.45555958988457506002e-5 + (0.23723907357214175198e-7 + (0.11839789326602695603e-9 + (0.56346163067550237877e-12 + 0.25403679644444444444e-14 * t) * t) * t) * t) * t) * t;
}
case 9: {
T t = 2*y100 - 19;
return 0.14800987015587535621e-1 + (0.86018092946345943214e-3 + (0.47008265848816866105e-5 + (0.24694040760197315333e-7 + (0.12418779768752299093e-9 + (0.59486890370320261949e-12 + 0.26957764568888888889e-14 * t) * t) * t) * t) * t) * t;
}
case 10: {
T t = 2*y100 - 21;
return 0.16540351739394069380e-1 + (0.87928458641241463952e-3 + (0.48520195793001753903e-5 + (0.25711774900881709176e-7 + (0.13030128534230822419e-9 + (0.62820097586874779402e-12 + 0.28612737351111111111e-14 * t) * t) * t) * t) * t) * t;
}
case 11: {
T t = 2*y100 - 23;
return 0.18318536789842392647e-1 + (0.89900542647891721692e-3 + (0.50094684089553365810e-5 + (0.26779777074218070482e-7 + (0.13675822186304615566e-9 + (0.66358287745352705725e-12 + 0.30375273884444444444e-14 * t) * t) * t) * t) * t) * t;
}
case 12: {
T t = 2*y100 - 25;
return 0.20136801964214276775e-1 + (0.91936908737673676012e-3 + (0.51734830914104276820e-5 + (0.27900878609710432673e-7 + (0.14357976402809042257e-9 + (0.70114790311043728387e-12 + 0.32252476000000000000e-14 * t) * t) * t) * t) * t) * t;
}
case 13: {
T t = 2*y100 - 27;
return 0.21996459598282740954e-1 + (0.94040248155366777784e-3 + (0.53443911508041164739e-5 + (0.29078085538049374673e-7 + (0.15078844500329731137e-9 + (0.74103813647499204269e-12 + 0.34251892320000000000e-14 * t) * t) * t) * t) * t) * t;
}
case 14: {
T t = 2*y100 - 29;
return 0.23898877187226319502e-1 + (0.96213386835900177540e-3 + (0.55225386998049012752e-5 + (0.30314589961047687059e-7 + (0.15840826497296335264e-9 + (0.78340500472414454395e-12 + 0.36381553564444444445e-14 * t) * t) * t) * t) * t) * t;
}
case 15: {
T t = 2*y100 - 31;
return 0.25845480155298518485e-1 + (0.98459293067820123389e-3 + (0.57082915920051843672e-5 + (0.31613782169164830118e-7 + (0.16646478745529630813e-9 + (0.82840985928785407942e-12 + 0.38649975768888888890e-14 * t) * t) * t) * t) * t) * t;
}
case 16: {
T t = 2*y100 - 33;
return 0.27837754783474696598e-1 + (0.10078108563256892757e-2 + (0.59020366493792212221e-5 + (0.32979263553246520417e-7 + (0.17498524159268458073e-9 + (0.87622459124842525110e-12 + 0.41066206488888888890e-14 * t) * t) * t) * t) * t) * t;
}
case 17: {
T t = 2*y100 - 35;
return 0.29877251304899307550e-1 + (0.10318204245057349310e-2 + (0.61041829697162055093e-5 + (0.34414860359542720579e-7 + (0.18399863072934089607e-9 + (0.92703227366365046533e-12 + 0.43639844053333333334e-14 * t) * t) * t) * t) * t) * t;
}
case 18: {
T t = 2*y100 - 37;
return 0.31965587178596443475e-1 + (0.10566560976716574401e-2 + (0.63151633192414586770e-5 + (0.35924638339521924242e-7 + (0.19353584758781174038e-9 + (0.98102783859889264382e-12 + 0.46381060817777777779e-14 * t) * t) * t) * t) * t) * t;
}
case 19: {
T t = 2*y100 - 39;
return 0.34104450552588334840e-1 + (0.10823541191350532574e-2 + (0.65354356159553934436e-5 + (0.37512918348533521149e-7 + (0.20362979635817883229e-9 + (0.10384187833037282363e-11 + 0.49300625262222222221e-14 * t) * t) * t) * t) * t) * t;
}
case 20: {
T t = 2*y100 - 41;
return 0.36295603928292425716e-1 + (0.11089526167995268200e-2 + (0.67654845095518363577e-5 + (0.39184292949913591646e-7 + (0.21431552202133775150e-9 + (0.10994259106646731797e-11 + 0.52409949102222222221e-14 * t) * t) * t) * t) * t) * t;
}
case 21: {
T t = 2*y100 - 43;
return 0.38540888038840509795e-1 + (0.11364917134175420009e-2 + (0.70058230641246312003e-5 + (0.40943644083718586939e-7 + (0.22563034723692881631e-9 + (0.11642841011361992885e-11 + 0.55721092871111111110e-14 * t) * t) * t) * t) * t) * t;
}
case 22: {
T t = 2*y100 - 45;
return 0.40842225954785960651e-1 + (0.11650136437945673891e-2 + (0.72569945502343006619e-5 + (0.42796161861855042273e-7 + (0.23761401711005024162e-9 + (0.12332431172381557035e-11 + 0.59246802364444444445e-14 * t) * t) * t) * t) * t) * t;
}
case 23: {
T t = 2*y100 - 47;
return 0.43201627431540222422e-1 + (0.11945628793917272199e-2 + (0.75195743532849206263e-5 + (0.44747364553960993492e-7 + (0.25030885216472953674e-9 + (0.13065684400300476484e-11 + 0.63000532853333333334e-14 * t) * t) * t) * t) * t) * t;
}
case 24: {
T t = 2*y100 - 49;
return 0.45621193513810471438e-1 + (0.12251862608067529503e-2 + (0.77941720055551920319e-5 + (0.46803119830954460212e-7 + (0.26375990983978426273e-9 + (0.13845421370977119765e-11 + 0.66996477404444444445e-14 * t) * t) * t) * t) * t) * t;
}
case 25: {
T t = 2*y100 - 51;
return 0.48103121413299865517e-1 + (0.12569331386432195113e-2 + (0.80814333496367673980e-5 + (0.48969667335682018324e-7 + (0.27801515481905748484e-9 + (0.14674637611609884208e-11 + 0.71249589351111111110e-14 * t) * t) * t) * t) * t) * t;
}
case 26: {
T t = 2*y100 - 53;
return 0.50649709676983338501e-1 + (0.12898555233099055810e-2 + (0.83820428414568799654e-5 + (0.51253642652551838659e-7 + (0.29312563849675507232e-9 + (0.15556512782814827846e-11 + 0.75775607822222222221e-14 * t) * t) * t) * t) * t) * t;
}
case 27: {
T t = 2*y100 - 55;
return 0.53263363664388864181e-1 + (0.13240082443256975769e-2 + (0.86967260015007658418e-5 + (0.53662102750396795566e-7 + (0.30914568786634796807e-9 + (0.16494420240828493176e-11 + 0.80591079644444444445e-14 * t) * t) * t) * t) * t) * t;
}
case 28: {
T t = 2*y100 - 57;
return 0.55946601353500013794e-1 + (0.13594491197408190706e-2 + (0.90262520233016380987e-5 + (0.56202552975056695376e-7 + (0.32613310410503135996e-9 + (0.17491936862246367398e-11 + 0.85713381688888888890e-14 * t) * t) * t) * t) * t) * t;
}
case 29: {
T t = 2*y100 - 59;
return 0.58702059496154081813e-1 + (0.13962391363223647892e-2 + (0.93714365487312784270e-5 + (0.58882975670265286526e-7 + (0.34414937110591753387e-9 + (0.18552853109751857859e-11 + 0.91160736711111111110e-14 * t) * t) * t) * t) * t) * t;
}
case 30: {
T t = 2*y100 - 61;
return 0.61532500145144778048e-1 + (0.14344426411912015247e-2 + (0.97331446201016809696e-5 + (0.61711860507347175097e-7 + (0.36325987418295300221e-9 + (0.19681183310134518232e-11 + 0.96952238400000000000e-14 * t) * t) * t) * t) * t) * t;
}
case 31: {
T t = 2*y100 - 63;
return 0.64440817576653297993e-1 + (0.14741275456383131151e-2 + (0.10112293819576437838e-4 + (0.64698236605933246196e-7 + (0.38353412915303665586e-9 + (0.20881176114385120186e-11 + 0.10310784480000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 32: {
T t = 2*y100 - 65;
return 0.67430045633130393282e-1 + (0.15153655418916540370e-2 + (0.10509857606888328667e-4 + (0.67851706529363332855e-7 + (0.40504602194811140006e-9 + (0.22157325110542534469e-11 + 0.10964842115555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 33: {
T t = 2*y100 - 67;
return 0.70503365513338850709e-1 + (0.15582323336495709827e-2 + (0.10926868866865231089e-4 + (0.71182482239613507542e-7 + (0.42787405890153386710e-9 + (0.23514379522274416437e-11 + 0.11659571751111111111e-13 * t) * t) * t) * t) * t) * t;
}
case 34: {
T t = 2*y100 - 69;
return 0.73664114037944596353e-1 + (0.16028078812438820413e-2 + (0.11364423678778207991e-4 + (0.74701423097423182009e-7 + (0.45210162777476488324e-9 + (0.24957355004088569134e-11 + 0.12397238257777777778e-13 * t) * t) * t) * t) * t) * t;
}
case 35: {
T t = 2*y100 - 71;
return 0.76915792420819562379e-1 + (0.16491766623447889354e-2 + (0.11823685320041302169e-4 + (0.78420075993781544386e-7 + (0.47781726956916478925e-9 + (0.26491544403815724749e-11 + 0.13180196462222222222e-13 * t) * t) * t) * t) * t) * t;
}
case 36: {
T t = 2*y100 - 73;
return 0.80262075578094612819e-1 + (0.16974279491709504117e-2 + (0.12305888517309891674e-4 + (0.82350717698979042290e-7 + (0.50511496109857113929e-9 + (0.28122528497626897696e-11 + 0.14010889635555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 37: {
T t = 2*y100 - 75;
return 0.83706822008980357446e-1 + (0.17476561032212656962e-2 + (0.12812343958540763368e-4 + (0.86506399515036435592e-7 + (0.53409440823869467453e-9 + (0.29856186620887555043e-11 + 0.14891851591111111111e-13 * t) * t) * t) * t) * t) * t;
}
case 38: {
T t = 2*y100 - 77;
return 0.87254084284461718231e-1 + (0.17999608886001962327e-2 + (0.13344443080089492218e-4 + (0.90900994316429008631e-7 + (0.56486134972616465316e-9 + (0.31698707080033956934e-11 + 0.15825697795555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 39: {
T t = 2*y100 - 79;
return 0.90908120182172748487e-1 + (0.18544478050657699758e-2 + (0.13903663143426120077e-4 + (0.95549246062549906177e-7 + (0.59752787125242054315e-9 + (0.33656597366099099413e-11 + 0.16815130613333333333e-13 * t) * t) * t) * t) * t) * t;
}
case 40: {
T t = 2*y100 - 81;
return 0.94673404508075481121e-1 + (0.19112284419887303347e-2 + (0.14491572616545004930e-4 + (0.10046682186333613697e-6 + (0.63221272959791000515e-9 + (0.35736693975589130818e-11 + 0.17862931591111111111e-13 * t) * t) * t) * t) * t) * t;
}
case 41: {
T t = 2*y100 - 83;
return 0.98554641648004456555e-1 + (0.19704208544725622126e-2 + (0.15109836875625443935e-4 + (0.10567036667675984067e-6 + (0.66904168640019354565e-9 + (0.37946171850824333014e-11 + 0.18971959040000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 42: {
T t = 2*y100 - 85;
return 0.10255677889470089531e0 + (0.20321499629472857418e-2 + (0.15760224242962179564e-4 + (0.11117756071353507391e-6 + (0.70814785110097658502e-9 + (0.40292553276632563925e-11 + 0.20145143075555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 43: {
T t = 2*y100 - 87;
return 0.10668502059865093318e0 + (0.20965479776148731610e-2 + (0.16444612377624983565e-4 + (0.11700717962026152749e-6 + (0.74967203250938418991e-9 + (0.42783716186085922176e-11 + 0.21385479360000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 44: {
T t = 2*y100 - 89;
return 0.11094484319386444474e0 + (0.21637548491908170841e-2 + (0.17164995035719657111e-4 + (0.12317915750735938089e-6 + (0.79376309831499633734e-9 + (0.45427901763106353914e-11 + 0.22696025653333333333e-13 * t) * t) * t) * t) * t) * t;
}
case 45: {
T t = 2*y100 - 91;
return 0.11534201115268804714e0 + (0.22339187474546420375e-2 + (0.17923489217504226813e-4 + (0.12971465288245997681e-6 + (0.84057834180389073587e-9 + (0.48233721206418027227e-11 + 0.24079890062222222222e-13 * t) * t) * t) * t) * t) * t;
}
case 46: {
T t = 2*y100 - 93;
return 0.11988259392684094740e0 + (0.23071965691918689601e-2 + (0.18722342718958935446e-4 + (0.13663611754337957520e-6 + (0.89028385488493287005e-9 + (0.51210161569225846701e-11 + 0.25540227111111111111e-13 * t) * t) * t) * t) * t) * t;
}
case 47: {
T t = 2*y100 - 95;
return 0.12457298393509812907e0 + (0.23837544771809575380e-2 + (0.19563942105711612475e-4 + (0.14396736847739470782e-6 + (0.94305490646459247016e-9 + (0.54366590583134218096e-11 + 0.27080225920000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 48: {
T t = 2*y100 - 97;
return 0.12941991566142438816e0 + (0.24637684719508859484e-2 + (0.20450821127475879816e-4 + (0.15173366280523906622e-6 + (0.99907632506389027739e-9 + (0.57712760311351625221e-11 + 0.28703099555555555556e-13 * t) * t) * t) * t) * t) * t;
}
case 49: {
T t = 2*y100 - 99;
return 0.13443048593088696613e0 + (0.25474249981080823877e-2 + (0.21385669591362915223e-4 + (0.15996177579900443030e-6 + (0.10585428844575134013e-8 + (0.61258809536787882989e-11 + 0.30412080142222222222e-13 * t) * t) * t) * t) * t) * t;
}
case 50: {
T t = 2*y100 - 101;
return 0.13961217543434561353e0 + (0.26349215871051761416e-2 + (0.22371342712572567744e-4 + (0.16868008199296822247e-6 + (0.11216596910444996246e-8 + (0.65015264753090890662e-11 + 0.32210394506666666666e-13 * t) * t) * t) * t) * t) * t;
}
case 51: {
T t = 2*y100 - 103;
return 0.14497287157673800690e0 + (0.27264675383982439814e-2 + (0.23410870961050950197e-4 + (0.17791863939526376477e-6 + (0.11886425714330958106e-8 + (0.68993039665054288034e-11 + 0.34101266222222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 52: {
T t = 2*y100 - 105;
return 0.15052089272774618151e0 + (0.28222846410136238008e-2 + (0.24507470422713397006e-4 + (0.18770927679626136909e-6 + (0.12597184587583370712e-8 + (0.73203433049229821618e-11 + 0.36087889048888888890e-13 * t) * t) * t) * t) * t) * t;
}
case 53: {
T t = 2*y100 - 107;
return 0.15626501395774612325e0 + (0.29226079376196624949e-2 + (0.25664553693768450545e-4 + (0.19808568415654461964e-6 + (0.13351257759815557897e-8 + (0.77658124891046760667e-11 + 0.38173420035555555555e-13 * t) * t) * t) * t) * t) * t;
}
case 54: {
T t = 2*y100 - 109;
return 0.16221449434620737567e0 + (0.30276865332726475672e-2 + (0.26885741326534564336e-4 + (0.20908350604346384143e-6 + (0.14151148144240728728e-8 + (0.82369170665974313027e-11 + 0.40360957457777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 55: {
T t = 2*y100 - 111;
return 0.16837910595412130659e0 + (0.31377844510793082301e-2 + (0.28174873844911175026e-4 + (0.22074043807045782387e-6 + (0.14999481055996090039e-8 + (0.87348993661930809254e-11 + 0.42653528977777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 56: {
T t = 2*y100 - 113;
return 0.17476916455659369953e0 + (0.32531815370903068316e-2 + (0.29536024347344364074e-4 + (0.23309632627767074202e-6 + (0.15899007843582444846e-8 + (0.92610375235427359475e-11 + 0.45054073102222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 57: {
T t = 2*y100 - 115;
return 0.18139556223643701364e0 + (0.33741744168096996041e-2 + (0.30973511714709500836e-4 + (0.24619326937592290996e-6 + (0.16852609412267750744e-8 + (0.98166442942854895573e-11 + 0.47565418097777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 58: {
T t = 2*y100 - 117;
return 0.18826980194443664549e0 + (0.35010775057740317997e-2 + (0.32491914440014267480e-4 + (0.26007572375886319028e-6 + (0.17863299617388376116e-8 + (0.10403065638343878679e-10 + 0.50190265831111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 59: {
T t = 2*y100 - 119;
return 0.19540403413693967350e0 + (0.36342240767211326315e-2 + (0.34096085096200907289e-4 + (0.27479061117017637474e-6 + (0.18934228504790032826e-8 + (0.11021679075323598664e-10 + 0.52931171733333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 60: {
T t = 2*y100 - 121;
return 0.20281109560651886959e0 + (0.37739673859323597060e-2 + (0.35791165457592409054e-4 + (0.29038742889416172404e-6 + (0.20068685374849001770e-8 + (0.11673891799578381999e-10 + 0.55790523093333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 61: {
T t = 2*y100 - 123;
return 0.21050455062669334978e0 + (0.39206818613925652425e-2 + (0.37582602289680101704e-4 + (0.30691836231886877385e-6 + (0.21270101645763677824e-8 + (0.12361138551062899455e-10 + 0.58770520160000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 62: {
T t = 2*y100 - 125;
return 0.21849873453703332479e0 + (0.40747643554689586041e-2 + (0.39476163820986711501e-4 + (0.32443839970139918836e-6 + (0.22542053491518680200e-8 + (0.13084879235290858490e-10 + 0.61873153262222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 63: {
T t = 2*y100 - 127;
return 0.22680879990043229327e0 + (0.42366354648628516935e-2 + (0.41477956909656896779e-4 + (0.34300544894502810002e-6 + (0.23888264229264067658e-8 + (0.13846596292818514601e-10 + 0.65100183751111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 64: {
T t = 2*y100 - 129;
return 0.23545076536988703937e0 + (0.44067409206365170888e-2 + (0.43594444916224700881e-4 + (0.36268045617760415178e-6 + (0.25312606430853202748e-8 + (0.14647791812837903061e-10 + 0.68453122631111111110e-13 * t) * t) * t) * t) * t) * t;
}
case 65: {
T t = 2*y100 - 131;
return 0.24444156740777432838e0 + (0.45855530511605787178e-2 + (0.45832466292683085475e-4 + (0.38352752590033030472e-6 + (0.26819103733055603460e-8 + (0.15489984390884756993e-10 + 0.71933206364444444445e-13 * t) * t) * t) * t) * t) * t;
}
case 66: {
T t = 2*y100 - 133;
return 0.25379911500634264643e0 + (0.47735723208650032167e-2 + (0.48199253896534185372e-4 + (0.40561404245564732314e-6 + (0.28411932320871165585e-8 + (0.16374705736458320149e-10 + 0.75541379822222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 67: {
T t = 2*y100 - 135;
return 0.26354234756393613032e0 + (0.49713289477083781266e-2 + (0.50702455036930367504e-4 + (0.42901079254268185722e-6 + (0.30095422058900481753e-8 + (0.17303497025347342498e-10 + 0.79278273368888888890e-13 * t) * t) * t) * t) * t) * t;
}
case 68: {
T t = 2*y100 - 137;
return 0.27369129607732343398e0 + (0.51793846023052643767e-2 + (0.53350152258326602629e-4 + (0.45379208848865015485e-6 + (0.31874057245814381257e-8 + (0.18277905010245111046e-10 + 0.83144182364444444445e-13 * t) * t) * t) * t) * t) * t;
}
case 69: {
T t = 2*y100 - 139;
return 0.28426714781640316172e0 + (0.53983341916695141966e-2 + (0.56150884865255810638e-4 + (0.48003589196494734238e-6 + (0.33752476967570796349e-8 + (0.19299477888083469086e-10 + 0.87139049137777777779e-13 * t) * t) * t) * t) * t) * t;
}
case 70: {
T t = 2*y100 - 141;
return 0.29529231465348519920e0 + (0.56288077305420795663e-2 + (0.59113671189913307427e-4 + (0.50782393781744840482e-6 + (0.35735475025851713168e-8 + (0.20369760937017070382e-10 + 0.91262442613333333334e-13 * t) * t) * t) * t) * t) * t;
}
case 71: {
T t = 2*y100 - 143;
return 0.30679050522528838613e0 + (0.58714723032745403331e-2 + (0.62248031602197686791e-4 + (0.53724185766200945789e-6 + (0.37827999418960232678e-8 + (0.21490291930444538307e-10 + 0.95513539182222222221e-13 * t) * t) * t) * t) * t) * t;
}
case 72: {
T t = 2*y100 - 145;
return 0.31878680111173319425e0 + (0.61270341192339103514e-2 + (0.65564012259707640976e-4 + (0.56837930287837738996e-6 + (0.40035151353392378882e-8 + (0.22662596341239294792e-10 + 0.99891109760000000000e-13 * t) * t) * t) * t) * t) * t;
}
case 73: {
T t = 2*y100 - 147;
return 0.33130773722152622027e0 + (0.63962406646798080903e-2 + (0.69072209592942396666e-4 + (0.60133006661885941812e-6 + (0.42362183765883466691e-8 + (0.23888182347073698382e-10 + 0.10439349811555555556e-12 * t) * t) * t) * t) * t) * t;
}
case 74: {
T t = 2*y100 - 149;
return 0.34438138658041336523e0 + (0.66798829540414007258e-2 + (0.72783795518603561144e-4 + (0.63619220443228800680e-6 + (0.44814499336514453364e-8 + (0.25168535651285475274e-10 + 0.10901861383111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 75: {
T t = 2*y100 - 151;
return 0.35803744972380175583e0 + (0.69787978834882685031e-2 + (0.76710543371454822497e-4 + (0.67306815308917386747e-6 + (0.47397647975845228205e-8 + (0.26505114141143050509e-10 + 0.11376390933333333333e-12 * t) * t) * t) * t) * t) * t;
}
case 76: {
T t = 2*y100 - 153;
return 0.37230734890119724188e0 + (0.72938706896461381003e-2 + (0.80864854542670714092e-4 + (0.71206484718062688779e-6 + (0.50117323769745883805e-8 + (0.27899342394100074165e-10 + 0.11862637614222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 77: {
T t = 2*y100 - 155;
return 0.38722432730555448223e0 + (0.76260375162549802745e-2 + (0.85259785810004603848e-4 + (0.75329383305171327677e-6 + (0.52979361368388119355e-8 + (0.29352606054164086709e-10 + 0.12360253370666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 78: {
T t = 2*y100 - 157;
return 0.40282355354616940667e0 + (0.79762880915029728079e-2 + (0.89909077342438246452e-4 + (0.79687137961956194579e-6 + (0.55989731807360403195e-8 + (0.30866246101464869050e-10 + 0.12868841946666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 79: {
T t = 2*y100 - 159;
return 0.41914223158913787649e0 + (0.83456685186950463538e-2 + (0.94827181359250161335e-4 + (0.84291858561783141014e-6 + (0.59154537751083485684e-8 + (0.32441553034347469291e-10 + 0.13387957943111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 80: {
T t = 2*y100 - 161;
return 0.43621971639463786896e0 + (0.87352841828289495773e-2 + (0.10002929142066799966e-3 + (0.89156148280219880024e-6 + (0.62480008150788597147e-8 + (0.34079760983458878910e-10 + 0.13917107176888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 81: {
T t = 2*y100 - 163;
return 0.45409763548534330981e0 + (0.91463027755548240654e-2 + (0.10553137232446167258e-3 + (0.94293113464638623798e-6 + (0.65972492312219959885e-8 + (0.35782041795476563662e-10 + 0.14455745872000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 82: {
T t = 2*y100 - 165;
return 0.47282001668512331468e0 + (0.95799574408860463394e-2 + (0.11135019058000067469e-3 + (0.99716373005509038080e-6 + (0.69638453369956970347e-8 + (0.37549499088161345850e-10 + 0.15003280712888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 83: {
T t = 2*y100 - 167;
return 0.49243342227179841649e0 + (0.10037550043909497071e-1 + (0.11750334542845234952e-3 + (0.10544006716188967172e-5 + (0.73484461168242224872e-8 + (0.39383162326435752965e-10 + 0.15559069118222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 84: {
T t = 2*y100 - 169;
return 0.51298708979209258326e0 + (0.10520454564612427224e-1 + (0.12400930037494996655e-3 + (0.11147886579371265246e-5 + (0.77517184550568711454e-8 + (0.41283980931872622611e-10 + 0.16122419680000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 85: {
T t = 2*y100 - 171;
return 0.53453307979101369843e0 + (0.11030120618800726938e-1 + (0.13088741519572269581e-3 + (0.11784797595374515432e-5 + (0.81743383063044825400e-8 + (0.43252818449517081051e-10 + 0.16692592640000000000e-12 * t) * t) * t) * t) * t) * t;
}
case 86: {
T t = 2*y100 - 173;
return 0.55712643071169299478e0 + (0.11568077107929735233e-1 + (0.13815797838036651289e-3 + (0.12456314879260904558e-5 + (0.86169898078969313597e-8 + (0.45290446811539652525e-10 + 0.17268801084444444444e-12 * t) * t) * t) * t) * t) * t;
}
case 87: {
T t = 2*y100 - 175;
return 0.58082532122519320968e0 + (0.12135935999503877077e-1 + (0.14584223996665838559e-3 + (0.13164068573095710742e-5 + (0.90803643355106020163e-8 + (0.47397540713124619155e-10 + 0.17850211608888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 88: {
T t = 2*y100 - 177;
return 0.60569124025293375554e0 + (0.12735396239525550361e-1 + (0.15396244472258863344e-3 + (0.13909744385382818253e-5 + (0.95651595032306228245e-8 + (0.49574672127669041550e-10 + 0.18435945564444444444e-12 * t) * t) * t) * t) * t) * t;
}
case 89: {
T t = 2*y100 - 179;
return 0.63178916494715716894e0 + (0.13368247798287030927e-1 + (0.16254186562762076141e-3 + (0.14695084048334056083e-5 + (0.10072078109604152350e-7 + (0.51822304995680707483e-10 + 0.19025081422222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 90: {
T t = 2*y100 - 181;
return 0.65918774689725319200e0 + (0.14036375850601992063e-1 + (0.17160483760259706354e-3 + (0.15521885688723188371e-5 + (0.10601827031535280590e-7 + (0.54140790105837520499e-10 + 0.19616655146666666667e-12 * t) * t) * t) * t) * t) * t;
}
case 91: {
T t = 2*y100 - 183;
return 0.68795950683174433822e0 + (0.14741765091365869084e-1 + (0.18117679143520433835e-3 + (0.16392004108230585213e-5 + (0.11155116068018043001e-7 + (0.56530360194925690374e-10 + 0.20209663662222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 92: {
T t = 2*y100 - 185;
return 0.71818103808729967036e0 + (0.15486504187117112279e-1 + (0.19128428784550923217e-3 + (0.17307350969359975848e-5 + (0.11732656736113607751e-7 + (0.58991125287563833603e-10 + 0.20803065333333333333e-12 * t) * t) * t) * t) * t) * t;
}
case 93: {
T t = 2*y100 - 187;
return 0.74993321911726254661e0 + (0.16272790364044783382e-1 + (0.20195505163377912645e-3 + (0.18269894883203346953e-5 + (0.12335161021630225535e-7 + (0.61523068312169087227e-10 + 0.21395783431111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 94: {
T t = 2*y100 - 189;
return 0.78330143531283492729e0 + (0.17102934132652429240e-1 + (0.21321800585063327041e-3 + (0.19281661395543913713e-5 + (0.12963340087354341574e-7 + (0.64126040998066348872e-10 + 0.21986708942222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 95: {
T t = 2*y100 - 191;
return 0.81837581041023811832e0 + (0.17979364149044223802e-1 + (0.22510330592753129006e-3 + (0.20344732868018175389e-5 + (0.13617902941839949718e-7 + (0.66799760083972474642e-10 + 0.22574701262222222222e-12 * t) * t) * t) * t) * t) * t;
}
case 96: {
T t = 2*y100 - 193;
return 0.85525144775685126237e0 + (0.18904632212547561026e-1 + (0.23764237370371255638e-3 + (0.21461248251306387979e-5 + (0.14299555071870523786e-7 + (0.69543803864694171934e-10 + 0.23158593688888888889e-12 * t) * t) * t) * t) * t) * t;
}
case 97: {
T t = 2*y100 - 195;
return 0.89402868170849933734e0 + (0.19881418399127202569e-1 + (0.25086793128395995798e-3 + (0.22633402747585233180e-5 + (0.15008997042116532283e-7 + (0.72357609075043941261e-10 + 0.23737194737777777778e-12 * t) * t) * t) * t) * t) * t;
}
case 98: {
T t = 2*y100 - 197;
return 0.93481333942870796363e0 + (0.20912536329780368893e-1 + (0.26481403465998477969e-3 + (0.23863447359754921676e-5 + (0.15746923065472184451e-7 + (0.75240468141720143653e-10 + 0.24309291271111111111e-12 * t) * t) * t) * t) * t) * t;
}
case 99: {
T t = 2*y100 - 199;
return 0.97771701335885035464e0 + (0.22000938572830479551e-1 + (0.27951610702682383001e-3 + (0.25153688325245314530e-5 + (0.16514019547822821453e-7 + (0.78191526829368231251e-10 + 0.24873652355555555556e-12 * t) * t) * t) * t) * t) * t;
}
  }
  // we only get here if y = 1, i.e. |x| < 4*eps, in which case
  // erfcx is within 1e-15 of 1..
  return 1.0;
}

template <typename T>
C10_HOST_DEVICE static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfcx(T x)
{
  if (at::_isnan(x)) {
    return x;
  }

  if (x >= 0) {
    if (x > 50) { // continued-fraction expansion is faster
      const T ispi = 0.56418958354775628694807945156; // 1 / sqrt(pi)
      if (x > 5e7) { // 1-term expansion, important to avoid overflow
        return ispi / x;
      }
      /* 5-term expansion (rely on compiler for CSE), simplified from:
                ispi / (x+0.5/(x+1/(x+1.5/(x+2/x))))  */
      return ispi*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75));
    }
    return erfcx_y100(400/(4+x));
  }
  else {
    if (x < -26.7) {
      return std::numeric_limits<T>::infinity();
    }
    else if (x < -6.1) {
      return 2*exp(x*x);
    }
    else {
      return 2*exp(x*x) - erfcx_y100(400/(4-x));
    }
  }
}

C10_CLANG_DIAGNOSTIC_POP()
