#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <torch/headeronly/util/complex.h>

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
