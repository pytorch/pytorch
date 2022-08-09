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
#include <ATen/native/special_functions/bessel_j_1.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/MathConstants.h>
#include <c10/util/math_compat.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T>
static inline C10_HOST_DEVICE
T bessel_y_1(T x) {
    static const T PP[] = {
            +7.62125616208173112003e-04,
            +7.31397056940917570436e-02,
            +1.12719608129684925192e+00,
            +5.11207951146807644818e+00,
            +8.42404590141772420927e+00,
            +5.21451598682361504063e+00,
            +1.00000000000000000254e+00,
    };

    static const T PQ[] = {
            +5.71323128072548699714e-04,
            +6.88455908754495404082e-02,
            +1.10514232634061696926e+00,
            +5.07386386128601488557e+00,
            +8.39985554327604159757e+00,
            +5.20982848682361821619e+00,
            +9.99999999999999997461e-01,
    };

    static const T QP[] = {
            +5.10862594750176621635e-02,
            +4.98213872951233449420e+00,
            +7.58238284132545283818e+01,
            +3.66779609360150777800e+02,
            +7.10856304998926107277e+02,
            +5.97489612400613639965e+02,
            +2.11688757100572135698e+02,
            +2.52070205858023719784e+01,
    };

    static const T QQ[] = {
            +7.42373277035675149943e+01,
            +1.05644886038262816351e+03,
            +4.98641058337653607651e+03,
            +9.56231892404756170795e+03,
            +7.99704160447350683650e+03,
            +2.82619278517639096600e+03,
            +3.36093607810698293419e+02,
    };

    static const T YP[] = {
            +1.26320474790178026440e+09,
            -6.47355876379160291031e+11,
            +1.14509511541823727583e+14,
            -8.12770255501325109621e+15,
            +2.02439475713594898196e+17,
            -7.78877196265950026825e+17,
    };

    static const T YQ[] = {
            +5.94301592346128195359e+02,
            +2.35564092943068577943e+05,
            +7.34811944459721705660e+07,
            +1.87601316108706159478e+10,
            +3.88231277496238566008e+12,
            +6.20557727146953693363e+14,
            +6.87141087355300489866e+16,
            +3.97270608116560655612e+18,
    };

    if (x <= T(5.0)) {
        if (x == T(0.0)) {
            return -std::numeric_limits<T>::infinity();
        }

        if (x <= T(0.0)) {
            return std::numeric_limits<T>::quiet_NaN();
        }

        T yp = 0.0;

        for (uint8_t index = 0; index <= 5; index++) {
            yp = yp * (x * x) + YP[index];
        }

        T yq = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            yq = yq * (x * x) + YQ[index];
        }

        return x * (yp / yq) + (T(0.636619772367581343075535053490057448) * (bessel_j_1(x) * std::log(x) - T(1.0) / x));
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

    return (pp / pq * std::sin(x - T(2.356194490192344928846982537459627163)) + T(5.0) / x * (qp / qq) * std::cos(x - T(2.356194490192344928846982537459627163))) * T(0.797884560802865355879892119868763737) / std::sqrt(x);
} // T bessel_y_1(T x)
} // namespace special_functions
} // namespace native
} // namespace at
