#ifndef THZ_AVX_H
#define THZ_AVX_H

#include <stddef.h>
#include "General.h"

void THZDoubleVector_copy_AVX(double _Complex *y, const double _Complex *x, const ptrdiff_t n);
void THZDoubleVector_fill_AVX(double _Complex *z, const double _Complex c, const ptrdiff_t n);
void THZDoubleVector_adds_AVX(double _Complex *y, const double _Complex *x, const double _Complex c, const ptrdiff_t n);

void THZFloatVector_copy_AVX(float _Complex *y, const float _Complex *x, const ptrdiff_t n);
void THZFloatVector_fill_AVX(float _Complex *z, const float _Complex c, const ptrdiff_t n);
void THZFloatVector_adds_AVX(float _Complex *y, const float _Complex *x, const float _Complex c, const ptrdiff_t n);

#endif
