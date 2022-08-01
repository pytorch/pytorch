#pragma once

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
