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
T bessel_j_1(T x) {
    static const T PP[] = {
            +7.62125616208173112003e-04, +7.31397056940917570436e-02,
            +1.12719608129684925192e+00, +5.11207951146807644818e+00,
            +8.42404590141772420927e+00, +5.21451598682361504063e+00,
            +1.00000000000000000254e+00,
    };

    static const T PQ[] = {
            +5.71323128072548699714e-04, +6.88455908754495404082e-02,
            +1.10514232634061696926e+00, +5.07386386128601488557e+00,
            +8.39985554327604159757e+00, +5.20982848682361821619e+00,
            +9.99999999999999997461e-01,
    };

    static const T QP[] = {
            +5.10862594750176621635e-02, +4.98213872951233449420e+00,
            +7.58238284132545283818e+01, +3.66779609360150777800e+02,
            +7.10856304998926107277e+02, +5.97489612400613639965e+02,
            +2.11688757100572135698e+02, +2.52070205858023719784e+01,
    };

    static const T QQ[] = {
            +7.42373277035675149943e+01, +1.05644886038262816351e+03,
            +4.98641058337653607651e+03, +9.56231892404756170795e+03,
            +7.99704160447350683650e+03, +2.82619278517639096600e+03,
            +3.36093607810698293419e+02,
    };

    static const T RP[] = {
            -8.99971225705559398224e+08, +4.52228297998194034323e+11,
            -7.27494245221818276015e+13, +3.68295732863852883286e+15,
    };

    static const T RQ[] = {
            +6.20836478118054335476e+02, +2.56987256757748830383e+05,
            +8.35146791431949253037e+07, +2.21511595479792499675e+10,
            +4.74914122079991414898e+12, +7.84369607876235854894e+14,
            +8.95222336184627338078e+16, +5.32278620332680085395e+18,
    };

    if (x < T(0.0)) {
        return -bessel_j_1(-x);
    }

    if (x <= T(5.0)) {
        T rp = 0.0;

        for (uint8_t index = 0; index <= 3; index++) {
            rp = rp * (x * x) + RP[index];
        }

        T rq = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            rq = rq * (x * x) + RQ[index];
        }

        return rp / rq * x * (x * x - T(1.46819706421238932572e+01)) * (x * x - T(4.92184563216946036703e+01));
    }

    T pp = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(5.0) / x * (T(5.0) / x)) + PP[index];
    }

    T pq = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(5.0) / x * (T(5.0) / x)) + PQ[index];
    }

    T qp = 0.0;

    for (uint8_t index = 0; index <= 7; index++) {
        qp = qp * (T(5.0) / x * (T(5.0) / x)) + QP[index];
    }

    T qq = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        qq = qq * (T(5.0) / x * (T(5.0) / x)) + QQ[index];
    }

    return (pp / pq * std::cos(x - T(2.356194490192344928846982537459627163)) - T(5.0) / x * (qp / qq) * std::sin(x - T(2.356194490192344928846982537459627163))) * T(0.797884560802865355879892119868763737) / std::sqrt(x);
} // T bessel_j_1(T x)
} // namespace special_functions
} // namespace native
} // namespace at
