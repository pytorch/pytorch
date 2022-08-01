#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T chebyshev_polynomial_u(T x, int64_t n) {
    if (n < 0) {
        return T(0.0);
    }

    if (std::abs(x) == T(1.0)) {
        if (x > T(0.0) || n % 2 == 0) {
            return n + 1;
        }

        return -(n + 1);
    }

    if ((n > 8) && (std::abs(x) < T(1.0))) {
        if (std::sin(std::acos(x)) != T(0.0)) {
            return std::sin((n + 1) * std::acos(x)) / std::sin(std::acos(x));
        }

        return (n + 1) * std::cos((n + 1) * std::acos(x)) / x;
    }

    if (n == 0) {
        return T(1.0);
    }

    if (n == 1) {
        return x + x;
    }

    T p = T(1.0);
    T q = x + x;
    T r;

    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    return r;
} // chebyshev_polynomial_u(T x, int64_t n)

template<typename T, bool is_cuda=false>
static inline C10_HOST_DEVICE
T chebyshev_polynomial_u(T x, T n) {
    return chebyshev_polynomial_u(x, static_cast<int64_t>(n));
} // T chebyshev_polynomial_u(T x, T n)
} // namespace special_functions
} // namespace native
} // namespace at
