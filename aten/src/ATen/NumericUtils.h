#pragma once

#include <c10/util/complex.h>

#include <cmath>
#include <type_traits>

#include <torch/headeronly/util/NumericUtils.h>

namespace at {

using torch::headeronly::_isinf;
using torch::headeronly::_isnan;
using torch::headeronly::exp;
using torch::headeronly::log;
using torch::headeronly::log1p;
using torch::headeronly::tan;

} // namespace at
