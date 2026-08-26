#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <torch/headeronly/util/complex.h>

// std::real/imag/abs/arg/norm/conj/exp for c10::complex now live in
// torch/headeronly/util/complex.h (included above), since they're fully
// inline and needed by the header-only ATen/native/Math.h and
// ATen/native/cpu/zmath.h. The remaining complex math functions (log, pow,
// trig/hyperbolic functions, etc.) stay here / in complex_math.h.

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H

namespace c10 {
using torch::headeronly::is_complex;
using torch::headeronly::scalar_value_type;
} // namespace c10
