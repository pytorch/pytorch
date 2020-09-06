#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <c10/util/math_compat.h>

#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif  // M_PIf

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
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(M_PI)))*std::exp(-x*x));
  x = x - (std::erf(x) - y) / ((static_cast<T>(2.0)/static_cast<T>(std::sqrt(M_PI)))*std::exp(-x*x));

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

static inline double trigamma(double x) {
  double sign = +1;
  double result = 0;
  if (x < 0.5) {
    sign = -1;
    const double sin_pi_x = sin(M_PI * x);
    result -= (M_PI * M_PI) / (sin_pi_x * sin_pi_x);
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

static inline float trigamma(float x) {
  float sign = +1;
  float result = 0;
  if (x < 0.5f) {
    sign = -1;
    const float sin_pi_x = sinf(M_PIf * x);
    result -= (M_PIf * M_PIf) / (sin_pi_x * sin_pi_x);
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
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }

  int x_is_integer = x == floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    return calc_digamma(1 - x) - M_PI / tan(M_PI * x);
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
  static float PSI_10 = 2.25175258906672110764f;
  if (x == 0) {
    return INFINITY;
  }

  int x_is_integer = x == floorf(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    // Avoid rounding errors for `tan`'s input.
    // Those make a big difference at extreme values.
    float pi_over_tan_pi_x = (float)(M_PI / tan(M_PI * (double)x));
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

inline c10::BFloat16 calc_erfinv(c10::BFloat16 a) { return calc_erfinv(float(a)); }

template <typename T>
static T abs_impl(T v) {
  return std::abs(v);
}

template <>
uint8_t abs_impl(uint8_t v) {
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
 chbevl(T x, T array[], size_t len) {
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
static inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_i0(T _x) {
  T x = std::abs(_x);
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static T A[] = {
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
  };

  /* Chebyshev coefficients for exp(-x) sqrt(x) I0(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I0(x) } = 1/sqrt(2pi).
   */
  static T B[] = {
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
  };

  if (x <= 8.0) {
    T y = (x / 2.0) - 2.0;
    return static_cast<T>(std::exp(x) * chbevl(y, A, 30));
  }

  return static_cast<T>(std::exp(x) * chbevl(static_cast<T>(32.0 / x - 2.0), B, 25) / std::sqrt(x));
}

// Upcast bfloat16 input to float for numerical accuracy purposes
inline c10::BFloat16 calc_i0(c10::BFloat16 a) { return calc_i0(static_cast<float>(a)); }
