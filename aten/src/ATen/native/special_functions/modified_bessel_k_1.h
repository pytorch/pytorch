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
#include <ATen/native/special_functions/modified_bessel_i_1.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T modified_bessel_k_1(T x) {
    static const T A[] = {
            -7.02386347938628759343e-18,
            -2.42744985051936593393e-15,
            -6.66690169419932900609e-13,
            -1.41148839263352776110e-10,
            -2.21338763073472585583e-08,
            -2.43340614156596823496e-06,
            -1.73028895751305206302e-04,
            -6.97572385963986435018e-03,
            -1.22611180822657148235e-01,
            -3.53155960776544875667e-01,
            +1.52530022733894777053e+00,
    };

    static const T B[] = {
            -5.75674448366501715755e-18,
            +1.79405087314755922667e-17,
            -5.68946255844285935196e-17,
            +1.83809354436663880070e-16,
            -6.05704724837331885336e-16,
            +2.03870316562433424052e-15,
            -7.01983709041831346144e-15,
            +2.47715442448130437068e-14,
            -8.97670518232499435011e-14,
            +3.34841966607842919884e-13,
            -1.28917396095102890680e-12,
            +5.13963967348173025100e-12,
            -2.12996783842756842877e-11,
            +9.21831518760500529508e-11,
            -4.19035475934189648750e-10,
            +2.01504975519703286596e-09,
            -1.03457624656780970260e-08,
            +5.74108412545004946722e-08,
            -3.50196060308781257119e-07,
            +2.40648494783721712015e-06,
            -1.93619797416608296024e-05,
            +1.95215518471351631108e-04,
            -2.85781685962277938680e-03,
            +1.03923736576817238437e-01,
            +2.72062619048444266945e+00,
    };

    if (x == T(0.0)) {
        return std::numeric_limits<T>::infinity();
    }

    if (x < T(0.0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T p;
    T q = 0.0;

    if (x <= T(2.0)) {
        T a = A[0];

        for (uint8_t index = 1; index < 11; index++) {
            p = q;
            q = a;
            a = (x * x - T(2.0)) * q - p + A[index];
        }

        return std::log(T(0.5) * x) * modified_bessel_i_1(x) + T(0.5) * (a - p) / x;
    }

    T b = B[0];

    for (uint8_t index = 1; index < 25; index++) {
        p = q;
        q = b;
        b = (T(8.0) / x - T(2.0)) * q - p + B[index];
    }

    return std::exp(-x) * (T(0.5) * (b - p)) / std::sqrt(x);
} // T modified_bessel_k_1(T x)
} // namespace special_functions
} // namespace native
} // namespace at
