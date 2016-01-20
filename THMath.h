#ifndef _THMATH_H
#define _THMATH_H

static inline double TH_sigmoid(double value) {
  return 1.0 / (1.0 + exp(-value));
}

#endif // _THMATH_H

