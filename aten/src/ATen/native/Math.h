#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/MathConstants.h>
#include <c10/util/math_compat.h>


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
static inline double zeta(double x, double q) {
  static double MACHEP = 1.11022302462515654042E-16;
  static double A[] = {
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
  double a, b, k, s, t, w;
  if (x == 1.0) {
    return INFINITY;
  }

  if (x < 1.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (q <= 0.0) {
    if (q == floor(q)) {
      return INFINITY;
    }
    if (x != floor(x)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  s = std::pow(q, -x);
  a = q;
  i = 0;
  b = 0.0;
  while ((i < 9) || (a <= 9.0)) {
    i += 1;
    a += 1.0;
    b = std::pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return s;
    }
  };

  w = a;
  s += b * w / (x - 1.0);
  s -= 0.5 * b;
  a = 1.0;
  k = 0.0;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / A[i];
    s = s + t;
    t = std::abs(t / s);
    if (t < MACHEP) {
      return s;
    }
    k += 1.0;
    a *= x + k;
    b /= w;
    k += 1.0;
  }
  return s;
}

static inline double polevl(double x, double *A, size_t len) {
  double result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

static inline float polevlf(float x, float *A, size_t len) {
  float result = 0;
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
  static double A[] = {
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
  static float A[] = {
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
    y = z * polevlf(z, A, 6);
  }
  return result + logf(x) - (0.5f / x) - y;
}

static inline double calc_polygamma(int64_t n, double x) {
  // already blocked if n <= 1
  return ((n % 2) ? 1.0 : -1.0) * std::exp(lgamma(double(n) + 1.0)) *
      zeta(double(n + 1), x);
}

static inline float calc_polygamma(int64_t n, float x) {
  // already blocked if n <= 1
  return ((n % 2) ? 1.0f : -1.0f) * std::exp(lgamma(double(n) + 1.0)) *
      zeta(double(n + 1), x);
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
 chbevl(const T x,const T array[], size_t len) {
  T b0, b1, b2;

  b0 = array[0];
  b1 = static_cast<T>(0.0);

  for (size_t i = 1; i < len; ++i)  {
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
inline const T* chebyshev_coefficients_A(){
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
  return coeff;
};

template <typename T>
inline const T* chebyshev_coefficients_B(){
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

  return coeff;
};

template <typename T>
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i0(T _x) {
  T x = std::abs(_x);

  if (x <= 8.0) {
    const auto A = chebyshev_coefficients_A<T>();
    T y = (x / 2.0) - 2.0;
    return static_cast<T>(std::exp(x) * chbevl(y, A, 30));
  }

  const auto B = chebyshev_coefficients_B<T>();
  return static_cast<T>(std::exp(x) * chbevl(static_cast<T>(32.0 / x - 2.0), B, 25) / std::sqrt(x));
}

// Upcast bfloat16 input to float for numerical accuracy purposes
inline c10::BFloat16 calc_i0(c10::BFloat16 a) { return calc_i0(static_cast<float>(a)); }

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

  if (x <= 8.0) {
    auto A = chebyshev_coefficients_A<T>();
    T y = (x / 2.0) - 2.0;
    return static_cast<T>(chbevl(y, A, 30));
  }

  auto B = chebyshev_coefficients_B<T>();
  return static_cast<T>(
      chbevl(static_cast<T>(32.0 / x - 2.0), B, 25) / std::sqrt(x));
}

// Upcast bfloat16 input to float for numerical accuracy purposes
inline c10::BFloat16 calc_i0e(c10::BFloat16 a) { return calc_i0e(static_cast<float>(a)); }
