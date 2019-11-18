#ifndef _THMATH_H
#define _THMATH_H
#include <stdlib.h>
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#ifndef M_PIf
#define M_PIf 3.1415926535f
#endif  // M_PIf

static inline double TH_sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

static inline float TH_sigmoidf(float value) {
  return 1.0f / (1.0f + expf(-value));
}

#endif // _THMATH_H
