#include <stdlib.h>
#include <math.h>

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
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
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
 * The following function comes with the following copyright notice.
 * It has been released under the BSD license.
 *
 * Cephes Math Library Release 2.8:  June, 2000
 * Copyright 1984, 1987, 1992, 2000 by Stephen L. Moshier
 */
static inline double calc_digamma(float x) {
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
  if (x < 1.0e17) {
    float z = 1 / (x * x);
    y = z * polevlf(z, A, 6);
  }
  return result + logf(x) - (0.5 / x) - y;
}