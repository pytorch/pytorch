#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T chebyshev_polynomial_w(T x, int64_t n) {
    if (n < 0) {
        return T(0.0);
    }

    if (std::abs(x) == T(1.0)) {
        if (x > T(0.0)) {
            return n + n + 1;
        }

        if (n % 2 == 0) {
            return T(1.0);
        }

        return T(-1.0);
    }

    if ((n > 8) && (std::abs(x) < T(1.0))) {
        if (std::cos(std::acos(x) / T(2.0)) != T(1.0)) {
            return std::sin((n + T(0.5)) * std::acos(x)) / std::sin(std::acos(x) / T(2.0));
        }

        if (x > T(0.0)) {
            return n + n + 1;
        }

        if (n % 2 == 0) {
            return T(1.0);
        }

        return T(-1.0);
    }

    if (n == 0) {
        return T(1.0);
    }

    if (n == 1) {
        return x + x + T(1.0);
    }

    T p = T(1.0);
    T q = x + x + T(1.0);
    T r;

    for (int64_t k = 2; k <= n; k++) {
        r = (x + x) * q - p;
        p = q;
        q = r;
    }

    return r;
} // chebyshev_polynomial_w(T x, int64_t n)

template<typename T, bool is_cuda=false>
static inline C10_HOST_DEVICE
T chebyshev_polynomial_w(T x, T n) {
    return chebyshev_polynomial_w(x, static_cast<int64_t>(n));
} // T chebyshev_polynomial_w(T x, T n)
} // namespace special_functions
} // namespace native
} // namespace at
