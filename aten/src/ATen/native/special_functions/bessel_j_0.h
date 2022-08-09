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
T bessel_j_0(T x) {
    static const T PP[] = {
            +7.96936729297347051624e-04, +8.28352392107440799803e-02,
            +1.23953371646414299388e+00, +5.44725003058768775090e+00,
            +8.74716500199817011941e+00, +5.30324038235394892183e+00,
            +9.99999999999999997821e-01,
    };

    static const T PQ[] = {
            +9.24408810558863637013e-04, +8.56288474354474431428e-02,
            +1.25352743901058953537e+00, +5.47097740330417105182e+00,
            +8.76190883237069594232e+00, +5.30605288235394617618e+00,
            +1.00000000000000000218e+00,
    };

    static const T QP[] = {
            -1.13663838898469149931e-02, -1.28252718670509318512e+00,
            -1.95539544257735972385e+01, -9.32060152123768231369e+01,
            -1.77681167980488050595e+02, -1.47077505154951170175e+02,
            -5.14105326766599330220e+01, -6.05014350600728481186e+00,
    };

    static const T QQ[] = {
            +6.43178256118178023184e+01, +8.56430025976980587198e+02,
            +3.88240183605401609683e+03, +7.24046774195652478189e+03,
            +5.93072701187316984827e+03, +2.06209331660327847417e+03,
            +2.42005740240291393179e+02,
    };

    static const T RP[] = {
            -4.79443220978201773821e+09, +1.95617491946556577543e+12,
            -2.49248344360967716204e+14, +9.70862251047306323952e+15,
    };

    static const T RQ[] = {
            +4.99563147152651017219e+02, +1.73785401676374683123e+05,
            +4.84409658339962045305e+07, +1.11855537045356834862e+10,
            +2.11277520115489217587e+12, +3.10518229857422583814e+14,
            +3.18121955943204943306e+16, +1.71086294081043136091e+18,
    };

    if (x < T(0)) {
        x = -x;
    }

    if (x <= T(5.0)) {
        if (x < T(0.00001)) {
            return T(1.0) - x * x / T(4.0);
        }

        T rp = 0.0;

        for (uint8_t index = 0; index <= 3; index++) {
            rp = rp * (x * x) + RP[index];
        }

        T rq = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            rq = rq * (x * x) + RQ[index];
        }

        return (x * x - T(5.78318596294678452118e+00)) * (x * x - T(3.04712623436620863991e+01)) * rp / rq;
    }

    T pp = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        pp = pp * (T(25.0) / (x * x)) + PP[index];
    }

    T pq = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        pq = pq * (T(25.0) / (x * x)) + PQ[index];
    }

    T qp = 0.0;

    for (uint8_t index = 0; index <= 7; index++) {
        qp = qp * (T(25.0) / (x * x)) + QP[index];
    }

    T qq = 0.0;

    for (uint8_t index = 0; index <= 6; index++) {
        qq = qq * (T(25.0) / (x * x)) + QQ[index];
    }

    return (pp / pq * std::cos(x - T(0.785398163397448309615660845819875721)) - T(5.0) / x * (qp / qq) * std::sin(x - T(0.785398163397448309615660845819875721))) * T(0.797884560802865355879892119868763737) / std::sqrt(x);
} // T bessel_j_0(T x)
} // namespace special_functions
} // namespace native
} // namespace at
