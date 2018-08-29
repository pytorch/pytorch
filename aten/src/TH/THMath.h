#ifndef _THMATH_H
#define _THMATH_H
#include <stdlib.h>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#include <float.h>
#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif  // M_PIf

static inline double TH_sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

static inline double TH_frac(double x) {
  return x - trunc(x);
}

static inline double TH_rsqrt(double x) {
  return 1.0 / sqrt(x);
}

static inline double TH_lerp(double a, double b, double weight) {
  return a + weight * (b-a);
}

static inline float TH_sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

static inline float TH_fracf(float x) {
  return x - truncf(x);
}

static inline float TH_rsqrtf(float x) {
  return 1.0f / sqrtf(x);
}

static inline float TH_lerpf(float a, float b, float weight) {
  return a + weight * (b-a);
}

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

static inline double TH_erfinv(double y) {
/* Function to calculate inverse error function.  Rational approximation
is used to generate an initial approximation, which is then improved to
full accuracy by two steps of Newton's method.  Code is a direct
translation of the erfinv m file in matlab version 2.0.
Author:  Gary L. Pavlis, Indiana University
Date:  February 1996
*/
    double x,z,num,dem; /*working variables */
    /* coefficients in rational expansion */
    double a[4]={ 0.886226899, -1.645349621,  0.914624893, -0.140543331};
    double b[4]={-2.118377725,  1.442710462, -0.329097515,  0.012229801};
    double c[4]={-1.970840454, -1.624906493,  3.429567803,  1.641345311};
    double d[2]={ 3.543889200,  1.637067800};
    if(fabs(y) > 1.0) return (atof("NaN"));  /* This needs IEEE constant*/
    if(fabs(y) == 1.0) return((copysign(1.0,y))*atof("INFINITY"));
    if(fabs(y) <= CENTRAL_RANGE){
            z = y*y;
            num = (((a[3]*z + a[2])*z + a[1])*z + a[0]);
            dem = ((((b[3]*z + b[2])*z + b[1])*z +b[0])*z + 1.0);
            x = y*num/dem;
    }
    else{
            z = sqrt(-log((1.0-fabs(y))/2.0));
            num = ((c[3]*z + c[2])*z + c[1])*z + c[0];
            dem = (d[1]*z + d[0])*z + 1.0;
            x = (copysign(1.0,y))*num/dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));
    x = x - (erf(x) - y)/( (2.0/sqrt(M_PI))*exp(-x*x));

    return(x);
}
#undef CENTRAL_RANGE

static inline double TH_polevl(double x, double *A, size_t len) {
  double result = 0;
  for (size_t i = 0; i <= len; i++) {
    result = result * x + A[i];
  }
  return result;
}

static inline float TH_polevlf(float x, float *A, size_t len) {
  float result = 0;
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
static inline double TH_digamma(double x) {
  static double PSI_10 = 2.25175258906672110764;
  if (x == 0) {
    return INFINITY;
  }

  int x_is_integer = x == floor(x);
  if (x < 0) {
    if (x_is_integer) {
      return INFINITY;
    }
    return TH_digamma(1 - x) - M_PI / tan(M_PI * x);
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
    y = z * TH_polevl(z, A, 6);
  }
  return result + log(x) - (0.5 / x) - y;
}

/*
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
static inline double TH_digammaf(float x) {
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
    return TH_digammaf(1 - x) - pi_over_tan_pi_x;
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
  if (x < 1.0e17) {
    float z = 1 / (x * x);
    y = z * TH_polevlf(z, A, 6);
  }
  return result + logf(x) - (0.5 / x) - y;
}

static inline double TH_trigamma(double x) {
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

static inline float TH_trigammaf(float x) {
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
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
static inline double chbevl(double x, double array[], int n)
{
  double b0, b1, b2, *p;
  int i;

  p = array;
  b0 = *p++;
  b1 = 0.0;
  i = n - 1;

  do {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + *p++;
  }
  while (--i);
  return (0.5 * (b0 - b2));
}

static inline float chbevlf(float x, float array[], int n)
{
  float b0, b1, b2, *p;
  int i;

  p = array;
  b0 = *p++;
  b1 = 0.0;
  i = n - 1;

  do {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + *p++;
  }
  while (--i);
  return (0.5 * (b0 - b2));
}


/*
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
static inline double TH_i0(double x){
  /* Chebyshev coefficients for exp(-x) I0(x)
   * in the interval [0,8].
   *
   * lim(x->0){ exp(-x) I0(x) } = 1.
   */
  static double I0_A[] = {
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
  static double I0_B[] = {
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
  if (x < 0){x = -x;}
  if (x <= 8.0) {
    const double y = (x / 2.0) - 2.0;
    return (exp(x) * chbevl(y, I0_A, 30));
  }
  return (exp(x) * chbevl(32.0 / x - 2.0, I0_B, 25) / sqrt(x));
}

static inline float TH_i0f(float x){
  static float I0_A[] = {
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
  static float I0_B[] = {
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
  if (x < 0){x = -x;}
  if (x <= 8.0) {
    const float y = (x / 2.0) - 2.0;
    return (exp(x) * chbevlf(y, I0_A, 30));
  }
  return (exp(x) * chbevlf(32.0 / x - 2.0, I0_B, 25) / sqrt(x));

}

static inline double TH_i1(double x){
  static double I1_A[] = {
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1
  };

  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static double I1_B[] = {
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1
  };
  double y, z;
  z = fabs(x);
  if (z <= 8.0) {
    y = (z / 2.0) - 2.0;
    z = chbevl(y, I1_A, 29) * z * exp(z);
  }
  else {
    z = exp(z) * chbevl(32.0 / z - 2.0, I1_B, 25) / sqrt(z);
  }
  if (x < 0.0){z = -z;}
  return (z);
}

static inline float TH_i1f(float x){
  static float I1_A[] = {
    2.77791411276104639959E-18,
    -2.11142121435816608115E-17,
    1.55363195773620046921E-16,
    -1.10559694773538630805E-15,
    7.60068429473540693410E-15,
    -5.04218550472791168711E-14,
    3.22379336594557470981E-13,
    -1.98397439776494371520E-12,
    1.17361862988909016308E-11,
    -6.66348972350202774223E-11,
    3.62559028155211703701E-10,
    -1.88724975172282928790E-9,
    9.38153738649577178388E-9,
    -4.44505912879632808065E-8,
    2.00329475355213526229E-7,
    -8.56872026469545474066E-7,
    3.47025130813767847674E-6,
    -1.32731636560394358279E-5,
    4.78156510755005422638E-5,
    -1.61760815825896745588E-4,
    5.12285956168575772895E-4,
    -1.51357245063125314899E-3,
    4.15642294431288815669E-3,
    -1.05640848946261981558E-2,
    2.47264490306265168283E-2,
    -5.29459812080949914269E-2,
    1.02643658689847095384E-1,
    -1.76416518357834055153E-1,
    2.52587186443633654823E-1
  };

  /* Chebyshev coefficients for exp(-x) sqrt(x) I1(x)
   * in the inverted interval [8,infinity].
   *
   * lim(x->inf){ exp(-x) sqrt(x) I1(x) } = 1/sqrt(2pi).
   */
  static float I1_B[] = {
    7.51729631084210481353E-18,
    4.41434832307170791151E-18,
    -4.65030536848935832153E-17,
    -3.20952592199342395980E-17,
    2.96262899764595013876E-16,
    3.30820231092092828324E-16,
    -1.88035477551078244854E-15,
    -3.81440307243700780478E-15,
    1.04202769841288027642E-14,
    4.27244001671195135429E-14,
    -2.10154184277266431302E-14,
    -4.08355111109219731823E-13,
    -7.19855177624590851209E-13,
    2.03562854414708950722E-12,
    1.41258074366137813316E-11,
    3.25260358301548823856E-11,
    -1.89749581235054123450E-11,
    -5.58974346219658380687E-10,
    -3.83538038596423702205E-9,
    -2.63146884688951950684E-8,
    -2.51223623787020892529E-7,
    -3.88256480887769039346E-6,
    -1.10588938762623716291E-4,
    -9.76109749136146840777E-3,
    7.78576235018280120474E-1
  };
  float y, z;
  z = fabs(x);
  if (z <= 8.0) {
    y = (z / 2.0) - 2.0;
    z = chbevlf(y, I1_A, 29) * z * exp(z);
  }
  else {
    z = exp(z) * chbevlf(32.0 / z - 2.0, I1_B, 25) / sqrt(z);
  }
  if (x < 0.0){z = -z;}
  return (z);
}

/*
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 *
 * Parts of the code are copyright:
 *
 *     Cephes Math Library Release 2.8:  June, 2000
 *     Copyright 1984, 1987, 1988, 2000 by Stephen L. Moshier
 *
 * And other parts:
 *
 *     Copyright (c) 2006 Xiaogang Zhang
 *     Use, modification and distribution are subject to the
 *     Boost Software License, Version 1.0.
 *
 *     Boost Software License - Version 1.0 - August 17th, 2003
 *
 *     Permission is hereby granted, free of charge, to any person or
 *     organization obtaining a copy of the software and accompanying
 *     documentation covered by this license (the "Software") to use, reproduce,
 *     display, distribute, execute, and transmit the Software, and to prepare
 *     derivative works of the Software, and to permit third-parties to whom the
 *     Software is furnished to do so, all subject to the following:
 *
 *     The copyright notices in the Software and this entire statement,
 *     including the above license grant, this restriction and the following
 *     disclaimer, must be included in all copies of the Software, in whole or
 *     in part, and all derivative works of the Software, unless such copies or
 *     derivative works are solely in the form of machine-executable object code
 *     generated by a source language processor.
 *
 *     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 *     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
 *     NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
 *     DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY,
 *     WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 *     CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *     SOFTWARE.
 *
 * And the rest are:
 *
 *     Copyright (C) 2009 Pauli Virtanen
 *     Distributed under the same license as Scipy.
 *
 */

#define MAXITER 2000

static const double MACHEP = 1.11022302462515654042E-16;
static double iv_asymptotic(double v, double x);
static void ikv_asymptotic_uniform(double v, double x, double *Iv, double *Kv);
static void ikv_temme(double v, double x, double *Iv, double *Kv);


static inline double TH_iv(double v, double x)
{
  int sign;
  double t, ax, res;

    /* If v is a negative integer, invoke symmetry */
  t = floor(v);
  if (v < 0.0) {
    if (t == v) {
        v = -v;     /* symmetry */
      t = -t;
    }
  }
    /* If x is negative, require v to be an integer */
  sign = 1;
  if (x < 0.0) {
    if (t != v) {
      THError("If x is negative, v needs to be an integer");
      return (NAN);
    }
    if (v != 2.0 * floor(v / 2.0)) {
      sign = -1;
    }
  }

    /* Avoid logarithm singularity */
  if (x == 0.0) {
    if (v == 0.0) {
      return 1.0;
    }
    if (v < 0.0) {
      THError("x can't be 0 unless v>=0");
      return INFINITY;
    }
    else
      return 0.0;
  }

  ax = fabs(x);
  if (fabs(v) > 50) {
    /*
     * Uniform asymptotic expansion for large orders.
     *
     * This appears to overflow slightly later than the Boost
     * implementation of Temme's method.
     */
    ikv_asymptotic_uniform(v, ax, &res, NULL);
  }
  else {
    /* Otherwise: Temme's method */
    ikv_temme(v, ax, &res, NULL);
  }
  res *= sign;
  return res;
}

static inline float TH_ivf(float vf, float xf)
{
  int sign;
  double t, ax, res, x, v ;
  float resf;
  x = (double) xf;
  v = (double) vf;
    /* If v is a negative integer, invoke symmetry */
  t = floor(v);
  if (v < 0.0) {
    if (t == v) {
        v = -v;     /* symmetry */
      t = -t;
    }
  }
    /* If x is negative, require v to be an integer */
  sign = 1;
  if (x < 0.0) {
    if (t != v) {
      THError("If x is negative, v needs to be an integer");
      return (NAN);
    }
    if (v != 2.0 * floor(v / 2.0)) {
      sign = -1;
    }
  }

    /* Avoid logarithm singularity */
  if (x == 0.0) {
    if (v == 0.0) {
      return 1.0;
    }
    if (v < 0.0) {
      THError("x can't be 0 unless v>=0");
      return INFINITY;
    }
    else
      return 0.0;
  }

  ax = fabs(x);
  if (fabs(v) > 50) {
    /*
     * Uniform asymptotic expansion for large orders.
     *
     * This appears to overflow slightly later than the Boost
     * implementation of Temme's method.
     */
    ikv_asymptotic_uniform(v, ax, &res, NULL);
  }
  else {
    /* Otherwise: Temme's method */
    ikv_temme(v, ax, &res, NULL);
  }
  res *= sign;
  resf = (float)res;
  return resf;
}

/*
 * Compute Iv from (AMS5 9.7.1), asymptotic expansion for large |z|
 * Iv ~ exp(x)/sqrt(2 pi x) ( 1 + (4*v*v-1)/8x + (4*v*v-1)(4*v*v-9)/8x/2! + ...)
 */
static double iv_asymptotic(double v, double x)
{
  double mu;
  double sum, term, prefactor, factor;
  int k;

  prefactor = exp(x) / sqrt(2 * M_PI * x);

  if (prefactor == INFINITY) {
    return prefactor;
  }

  mu = 4 * v * v;
  sum = 1.0;
  term = 1.0;
  k = 1;

  do {
    factor = (mu - (2 * k - 1) * (2 * k - 1)) / (8 * x) / k;
    if (k > 100) {
      /* didn't converge */
      #ifdef DEBUG
        THError("iv_asymptotic didn't converge in 100 iterations");
      #endif //DEBUG
      break;
    }
    term *= -factor;
    sum += term;
    ++k;
  } while (fabs(term) > MACHEP * fabs(sum));
  return sum * prefactor;
}


/*
 * Uniform asymptotic expansion factors, (AMS5 9.3.9; AMS5 9.3.10)
 *
 * Computed with:
 * --------------------
  import numpy as np
  t = np.poly1d([1,0])
  def up1(p):
  return .5*t*t*(1-t*t)*p.deriv() + 1/8. * ((1-5*t*t)*p).integ()
  us = [np.poly1d([1])]
  for k in range(10):
  us.append(up1(us[-1]))
  n = us[-1].order
  for p in us:
  print "{" + ", ".join(["0"]*(n-p.order) + map(repr, p)) + "},"
  print "N_UFACTORS", len(us)
  print "N_UFACTOR_TERMS", us[-1].order + 1
 * --------------------
 */
#define N_UFACTORS 11
#define N_UFACTOR_TERMS 31
static const double asymptotic_ufactors[N_UFACTORS][N_UFACTOR_TERMS] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, -0.20833333333333334, 0.0, 0.125, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0.3342013888888889, 0.0, -0.40104166666666669, 0.0, 0.0703125, 0.0,
     0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     -1.0258125964506173, 0.0, 1.8464626736111112, 0.0,
     -0.89121093750000002, 0.0, 0.0732421875, 0.0, 0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     4.6695844234262474, 0.0, -11.207002616222995, 0.0, 8.78912353515625,
     0.0, -2.3640869140624998, 0.0, 0.112152099609375, 0.0, 0.0, 0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28.212072558200244, 0.0,
     84.636217674600744, 0.0, -91.818241543240035, 0.0, 42.534998745388457,
     0.0, -7.3687943594796312, 0.0, 0.22710800170898438, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 212.5701300392171, 0.0,
     -765.25246814118157, 0.0, 1059.9904525279999, 0.0,
     -699.57962737613275, 0.0, 218.19051174421159, 0.0,
     -26.491430486951554, 0.0, 0.57250142097473145, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, -1919.4576623184068, 0.0,
     8061.7221817373083, 0.0, -13586.550006434136, 0.0, 11655.393336864536,
     0.0, -5305.6469786134048, 0.0, 1200.9029132163525, 0.0,
     -108.09091978839464, 0.0, 1.7277275025844574, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0},
    {0, 0, 0, 0, 0, 0, 20204.291330966149, 0.0, -96980.598388637503, 0.0,
     192547.0012325315, 0.0, -203400.17728041555, 0.0, 122200.46498301747,
     0.0, -41192.654968897557, 0.0, 7109.5143024893641, 0.0,
     -493.915304773088, 0.0, 6.074042001273483, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0},
    {0, 0, 0, -242919.18790055133, 0.0, 1311763.6146629769, 0.0,
     -2998015.9185381061, 0.0, 3763271.2976564039, 0.0,
     -2813563.2265865342, 0.0, 1268365.2733216248, 0.0,
     -331645.17248456361, 0.0, 45218.768981362737, 0.0,
     -2499.8304818112092, 0.0, 24.380529699556064, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0},
    {3284469.8530720375, 0.0, -19706819.11843222, 0.0, 50952602.492664628,
     0.0, -74105148.211532637, 0.0, 66344512.274729028, 0.0,
     -37567176.660763353, 0.0, 13288767.166421819, 0.0,
     -2785618.1280864552, 0.0, 308186.40461266245, 0.0,
     -13886.089753717039, 0.0, 110.01714026924674, 0.0, 0.0, 0.0, 0.0, 0.0,
     0.0, 0.0, 0.0, 0.0, 0.0}
};


/*
 * Compute Iv, Kv from (AMS5 9.7.7 + 9.7.8), asymptotic expansion for large v
 */
static void ikv_asymptotic_uniform(double v, double x, double *i_value, double *k_value)
{
  double i_prefactor, k_prefactor;
  double t, t2, eta, z;
  double i_sum, k_sum, term, divisor;
  int k, n;
  int sign = 1;

  if (v < 0) {
    /* Negative v; compute I_{-v} and K_{-v} and use (AMS 9.6.2) */
    sign = -1;
    v = -v;
  }

  z = x / v;
  t = 1 / sqrt(1 + z * z);
  t2 = t * t;
  eta = sqrt(1 + z * z) + log(z / (1 + 1 / t));

  i_prefactor = sqrt(t / (2 * M_PI * v)) * exp(v * eta);
  i_sum = 1.0;

  k_prefactor = sqrt(M_PI * t / (2 * v)) * exp(-v * eta);
  k_sum = 1.0;

  divisor = v;
  for (n = 1; n < N_UFACTORS; ++n) {
    /*
     * Evaluate u_k(t) with Horner's scheme;
     * (using the knowledge about which coefficients are zero)
     */
    term = 0;
    for (k = N_UFACTOR_TERMS - 1 - 3 * n; k < N_UFACTOR_TERMS - n; k += 2) {
      term *= t2;
      term += asymptotic_ufactors[n][k];
    }
    for (k = 1; k < n; k += 2) {
      term *= t2;
    }
    if (n % 2 == 1) {
      term *= t;
    }

    /* Sum terms */
    term /= divisor;
    i_sum += term;
    k_sum += (n % 2 == 0) ? term : -term;

    /* Check convergence */
    if (fabs(term) < MACHEP) {
      break;
    }

    divisor *= v;
  }
  #ifdef DEBUG
    if (fabs(term) > 1e-3 * fabs(i_sum)) {
        /* Didn't converge */
      THError("ikv_asymptotic_uniform didn't converge");
    }
    if (fabs(term) > MACHEP * fabs(i_sum)) {
        /* Some precision lost */
        THError("ikv_asymptotic_uniform didn't converge fully");
    }
  #endif //DEBUG
  if (k_value != NULL) {
      /* symmetric in v */
    *k_value = k_prefactor * k_sum;
  }

  if (i_value != NULL) {
    if (sign == 1) {
      *i_value = i_prefactor * i_sum;
    }
    else {
          /* (AMS 9.6.2) */
      *i_value = (i_prefactor * i_sum + (2 / M_PI) * sin(M_PI * v) * k_prefactor * k_sum);
    }
  }
}


/*
 * The following code originates from the Boost C++ library,
 * from file `boost/math/special_functions/detail/bessel_ik.hpp`,
 * converted from C++ to C.
 */

/*
 * Modified Bessel functions of the first and second kind of fractional order
 *
 * Calculate K(v, x) and K(v+1, x) by method analogous to
 * Temme, Journal of Computational Physics, vol 21, 343 (1976)
 */
static int temme_ik_series(double v, double x, double *K, double *K1)
{
  double f, h, p, q, coef, sum, sum1, tolerance;
  double a, b, c, d, sigma, gamma1, gamma2;
  unsigned long k;
  double gp;
  double gm;


    /*
     * |x| <= 2, Temme series converge rapidly
     * |x| > 2, the larger the |x|, the slower the convergence
     */

  gp = tgamma(v + 1) - 1;
  gm = tgamma(-v + 1) - 1;

  a = log(x / 2);
  b = exp(v * a);
  sigma = -a * v;
  c = fabs(v) < MACHEP ? 1 : sin(M_PI * v) / (v * M_PI);
  d = fabs(sigma) < MACHEP ? 1 : sinh(sigma) / sigma;
  double EULER = 0.577215664901532860606512090082402431; // Euler-Mascheroni constant
  gamma1 = fabs(v) < MACHEP ? - EULER : (0.5f / v) * (gp - gm) * c;
  gamma2 = (2 + gp + gm) * c / 2;

  /* initial values */
  p = (gp + 1) / (2 * b);
  q = (1 + gm) * b / 2;
  f = (cosh(sigma) * gamma1 + d * (-a) * gamma2) / c;
  h = p;
  coef = 1;
  sum = coef * f;
  sum1 = coef * h;

  /* series summation */
  tolerance = MACHEP;
  for (k = 1; k < MAXITER; k++) {
    f = (k * f + p + q) / (k * k - v * v);
    p /= k - v;
    q /= k + v;
    h = p - k * f;
    coef *= x * x / (4 * k);
    sum += coef * f;
    sum1 += coef * h;
    if (fabs(coef * f) < fabs(sum) * tolerance) {
      break;
    }
  }
  #ifdef DEBUG
    if (k == MAXITER) {
    THError("ikv_temme(temme_ik_series) TLOSS");
    }
  #endif //DEBUG

  *K = sum;
  *K1 = 2 * sum1 / x;

  return 0;
}

/* Evaluate continued fraction fv = I_(v+1) / I_v, derived from
 * Abramowitz and Stegun, Handbook of Mathematical Functions, 1972, 9.1.73 */
static int CF1_ik(double v, double x, double *fv)
{
  double C, D, f, a, b, delta, tiny, tolerance;
  unsigned long k;


    /*
     * |x| <= |v|, CF1_ik converges rapidly
     * |x| > |v|, CF1_ik needs O(|x|) iterations to converge
     */

    /*
     * modified Lentz's method, see
     * Lentz, Applied Optics, vol 15, 668 (1976)
     */
  tolerance = 2 * MACHEP;
  tiny = 1 / sqrt(DBL_MAX);
  C = f = tiny;       /* b0 = 0, replace with tiny */
  D = 0;
  for (k = 1; k < MAXITER; k++) {
    a = 1;
    b = 2 * (v + k) / x;
    C = b + a / C;
    D = b + a * D;
    if (C == 0) {
      C = tiny;
    }
    if (D == 0) {
      D = tiny;
    }
    D = 1 / D;
    delta = C * D;
    f *= delta;
    if (fabs(delta - 1) <= tolerance) {
      break;
    }
  }
  #ifdef DEBUG
    if (k == MAXITER) {
      THError("ikv_temme(CF1_ik) TLOSS");
    }
  #endif //DEBUG

  *fv = f;

  return 0;
}

/*
 * Calculate K(v, x) and K(v+1, x) by evaluating continued fraction
 * z1 / z0 = U(v+1.5, 2v+1, 2x) / U(v+0.5, 2v+1, 2x), see
 * Thompson and Barnett, Computer Physics Communications, vol 47, 245 (1987)
 */
static int CF2_ik(double v, double x, double *Kv, double *Kv1)
{

  double S, C, Q, D, f, a, b, q, delta, tolerance, current, prev;
  unsigned long k;

    /*
     * |x| >= |v|, CF2_ik converges rapidly
     * |x| -> 0, CF2_ik fails to converge
     */

    /*
     * Steed's algorithm, see Thompson and Barnett,
     * Journal of Computational Physics, vol 64, 490 (1986)
     */
  tolerance = MACHEP;
  a = v * v - 0.25f;
  b = 2 * (x + 1);        /* b1 */
  D = 1 / b;          /* D1 = 1 / b1 */
  f = delta = D;      /* f1 = delta1 = D1, coincidence */
  prev = 0;           /* q0 */
  current = 1;        /* q1 */
  Q = C = -a;         /* Q1 = C1 because q1 = 1 */
  S = 1 + Q * delta;      /* S1 */
  for (k = 2; k < MAXITER; k++) { /* starting from 2 */
    /* continued fraction f = z1 / z0 */
    a -= 2 * (k - 1);
    b += 2;
    D = 1 / (b + a * D);
    delta *= b * D - 1;
    f += delta;

      /* series summation S = 1 + \sum_{n=1}^{\infty} C_n * z_n / z_0 */
    q = (prev - (b - 2) * current) / a;
    prev = current;
      current = q;        /* forward recurrence for q */
    C *= -a / k;
    Q += C * q;
    S += Q * delta;

      /* S converges slower than f */
    if (fabs(Q * delta) < fabs(S) * tolerance) {
      break;
    }
  }
  #ifdef DEBUG
    if (k == MAXITER) {
      THError("ikv_temme(CF2_ik) TLOSS");
    }
  #endif //DEBUG

  *Kv = sqrt(M_PI / (2 * x)) * exp(-x) / S;
  *Kv1 = *Kv * (0.5f + v + x + (v * v - 0.25f) * f) / x;

  return 0;
}

/* Flags for what to compute */
enum {
    need_i = 0x1,
    need_k = 0x2
};

/*
 * Compute I(v, x) and K(v, x) simultaneously by Temme's method, see
 * Temme, Journal of Computational Physics, vol 19, 324 (1975)
 */
static void ikv_temme(double v, double x, double *Iv_p, double *Kv_p)
{
    /* Kv1 = K_(v+1), fv = I_(v+1) / I_v */
    /* Ku1 = K_(u+1), fu = I_(u+1) / I_u */
  double u, Iv, Kv, Kv1, Ku, Ku1, fv;
  double W, current, prev, next;
  int reflect = 0;
  unsigned n, k;
  int kind;

  kind = 0;
  if (Iv_p != NULL) {
    kind |= need_i;
  }
  if (Kv_p != NULL) {
    kind |= need_k;
  }

  if (v < 0) {
    reflect = 1;
    v = -v;         /* v is non-negative from here */
    kind |= need_k;
  }
  n = round(v);
  u = v - n;          /* -1/2 <= u < 1/2 */

  if (x < 0) {
    if (Iv_p != NULL)
      *Iv_p = NAN;
    if (Kv_p != NULL)
      *Kv_p = NAN;
    #ifdef DEBUG
        THError("ikv_temme DOMAIN");
    #endif //DEBUG
    return;
  }
  if (x == 0) {
    Iv = (v == 0) ? 1 : 0;
    if (kind & need_k) {
      #ifdef DEBUG
        THError("ikv_temme OVERFLOW");
      #endif //DEBUG
      Kv = INFINITY;
    }
    else {
      Kv = NAN;   /* any value will do */
    }

    if (reflect && (kind & need_i)) {
      double z = (u + n % 2);
      Iv = sin(M_PI * z) == 0 ? Iv : INFINITY;
      #ifdef DEBUG
        if (Iv == INFINITY || Iv == -INFINITY) {
          THError("ikv_temme OVERFLOW");
        }
      #endif //DEBUG
    }

    if (Iv_p != NULL) {
      *Iv_p = Iv;
    }
    if (Kv_p != NULL) {
      *Kv_p = Kv;
    }
    return;
  }
  /* x is positive until reflection */
  W = 1 / x;          /* Wronskian */
  if (x <= 2) {       /* x in (0, 2] */
    temme_ik_series(u, x, &Ku, &Ku1);   /* Temme series */
  }
  else {          /* x in (2, \infty) */
    CF2_ik(u, x, &Ku, &Ku1);    /* continued fraction CF2_ik */
  }
  prev = Ku;
  current = Ku1;
  for (k = 1; k <= n; k++) {  /* forward recurrence for K */
    next = 2 * (u + k) * current / x + prev;
    prev = current;
    current = next;
  }
  Kv = prev;
  Kv1 = current;
  if (kind & need_i) {
    double lim = (4 * v * v + 10) / (8 * x);

    lim *= lim;
    lim *= lim;
    lim /= 24;
    if ((lim < MACHEP * 10) && (x > 100)) {
          /*
           * x is huge compared to v, CF1 may be very slow
           * to converge so use asymptotic expansion for large
           * x case instead.  Note that the asymptotic expansion
           * isn't very accurate - so it's deliberately very hard
           * to get here - probably we're going to overflow:
           */
      Iv = iv_asymptotic(v, x);
    }
    else {
          CF1_ik(v, x, &fv);  /* continued fraction CF1_ik */
          Iv = W / (Kv * fv + Kv1);   /* Wronskian relation */
    }
  }
  else {
      Iv = NAN;       /* any value will do */
  }

  if (reflect) {
    double z = (u + n % 2);

    if (Iv_p != NULL) {
          *Iv_p = Iv + (2 / M_PI) * sin(M_PI * z) * Kv;   /* reflection formula */
    }
    if (Kv_p != NULL) {
      *Kv_p = Kv;
    }
  }
  else {
    if (Iv_p != NULL) {
      *Iv_p = Iv;
    }
    if (Kv_p != NULL) {
      *Kv_p = Kv;
    }
  }
  return;
}

#endif // _THMATH_H
