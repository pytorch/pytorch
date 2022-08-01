#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T hermite_polynomial_he(T x, int64_t n) {
    if (n < 0) {
        return T(0.0);
    }

    if (n == 0) {
        return T(1.0);
    }

    if (n == 1) {
        return x;
    }

    T p = T(1.0);
    T q = x;
    T r;

    for (int64_t k = 1; k < n; k++) {
        r = x * q - k * p;
        p = q;
        q = r;
    }

    return r;
} // hermite_polynomial_he(T x, int64_t n)

template<typename T, bool is_cuda=false>
static inline C10_HOST_DEVICE
T hermite_polynomial_he(T x, T n) {
    return hermite_polynomial_he(x, static_cast<int64_t>(n));
} // T hermite_polynomial_he(T x, T n)
} // namespace special_functions
} // namespace native
} // namespace at
