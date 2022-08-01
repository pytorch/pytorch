#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T laguerre_polynomial_l(T x, int64_t n) {
    if (n < 0) {
        return T(0.0);
    }

    if (std::abs(x) == T(0.0)) {
        return T(1.0);
    }

    if (n == 0) {
        return T(1.0);
    }

    if (n == 1) {
        return T(1.0) - x;
    }

    T p = T(1.0);
    T q = T(1.0) - x;
    T r;

    for (int64_t k = 1; k < n; k++) {
        r = (((k + k) + (T(1.0) - x)) * q - k * p) / (k + 1);
        p = q;
        q = r;
    }

    return r;
} // laguerre_polynomial_l(T x, int64_t n)

template<typename T, bool is_cuda=false>
static inline C10_HOST_DEVICE
T laguerre_polynomial_l(T x, T n) {
    return laguerre_polynomial_l(x, static_cast<int64_t>(n));
} // T laguerre_polynomial_l(T x, T n)
} // namespace special_functions
} // namespace native
} // namespace at
