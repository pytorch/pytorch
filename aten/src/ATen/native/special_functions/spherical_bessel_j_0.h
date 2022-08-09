#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include <ATen/AccumulateType.h>
#include <ATen/NumericUtils.h>
#include <ATen/jiterator_macros.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/MathConstants.h>
#include <c10/util/math_compat.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T spherical_bessel_j_0(T x) {
    if (std::isinf(x)) {
        return T(0.0);
    }

    if (std::abs(x) < T(0.5)) {
        return T(1.0) + x * x * (T(-1.0) / T(6.0) + x * x * (T(1.0) / T(120.0) + x * x * (T(-1.0) / T(5040.0) + x * x * (T(1.0) / T(362880.0) + x * x * (T(-1.0) / T(39916800.0) + x * x * (T(1.0) / T(6227020800.0)))))));
    }

    return std::sin(x) / x;
} // T spherical_bessel_j_0(T x)
} // namespace special_functions
} // namespace native
} // namespace at
