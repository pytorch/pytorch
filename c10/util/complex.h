#pragma once

#include <complex>

#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <torch/headeronly/util/complex.h>

namespace c10 {
using torch::headeronly::complex;
using torch::headeronly::operator!=;
using torch::headeronly::operator+;
using torch::headeronly::operator-;
using torch::headeronly::operator*;
using torch::headeronly::operator/;
using torch::headeronly::operator<<;
using torch::headeronly::operator>>;
using torch::headeronly::operator+=;
using torch::headeronly::operator-=;
using torch::headeronly::operator*=;
using torch::headeronly::operator/=;
using torch::headeronly::operator==;
using torch::headeronly::is_complex;
using torch::headeronly::polar;
using torch::headeronly::scalar_value_type;

namespace complex_literals {
using torch::headeronly::complex_literals::operator""_if;
using torch::headeronly::complex_literals::operator""_id;
} // namespace complex_literals
} // namespace c10

#define C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
// math functions are included in a separate file
#include <c10/util/complex_math.h> // IWYU pragma: keep
#undef C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H
