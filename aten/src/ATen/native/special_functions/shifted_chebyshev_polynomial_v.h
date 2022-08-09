#pragma once

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <ATen/NumericUtils.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/MathConstants.h>
#include <c10/util/math_compat.h>
#include <ATen/AccumulateType.h>
#include <ATen/jiterator_macros.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T shifted_chebyshev_polynomial_v(T x, int64_t n) {
    if (n < 0) {
        return T(0.0);
    }

    if (x == T(1.0)) {
        return T(1.0);
    }

    if (x == T(0.0)) {
        if (n % 2 == 0) {
            return (n + n + 1);
        }

        return -(n + n + 1);
    }

    if ((n > 6) && (std::abs(x + x - T(1.0)) < T(1.0))) {
        if (std::sin(std::acos(x + x - T(1.0)) / T(2.0)) != T(1.0)) {
            return std::cos(((n) + T(0.5)) * std::acos(x + x - T(1.0))) / std::cos(std::acos(x + x - T(1.0)) / T(2.0));
        }

        if (n % 2 == 0) {
            return n + n + 1;
        }

        return -(n + n + 1);
    }

    if (n == 0) {
        return T(1.0);
    }

    if (n == 1) {
        return x + x - T(1.0) + (x + x - T(1.0)) - T(1.0);
    }

    T p = T(1.0);
    T q = x + x - T(1.0) + (x + x - T(1.0)) - T(1.0);
    T r;

    for (int64_t k = 2; k <= n; k++) {
        r = (x + x - T(1.0) + (x + x - T(1.0))) * q - p;
        p = q;
        q = r;
    }

    return r;
} // shifted_chebyshev_polynomial_v(T x, int64_t n)

template<typename T, bool is_cuda=false>
static inline C10_HOST_DEVICE
T shifted_chebyshev_polynomial_v(T x, T n) {
    return shifted_chebyshev_polynomial_v(x, static_cast<int64_t>(n));
} // T shifted_chebyshev_polynomial_v(T x, T n)
} // namespace special_functions
} // namespace native
} // namespace at
