#ifndef _THMATH_H
#define _THMATH_H

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

#endif // _THMATH_H

