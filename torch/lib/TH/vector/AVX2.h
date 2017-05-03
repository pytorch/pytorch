#ifndef TH_AVX2_H
#define TH_AVX2_H

#include <stddef.h>

void THDoubleVector_cadd_AVX2(double *z, const double *x, const double *y, const double c, const ptrdiff_t n);
void THFloatVector_cadd_AVX2(float *z, const float *x, const float *y, const float c, const ptrdiff_t n);

#endif
