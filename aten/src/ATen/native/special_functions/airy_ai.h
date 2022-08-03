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
T airy_ai(T x) {
    static const T AN[] = {
            +3.46538101525629032477e-01,
            +1.20075952739645805542e+01,
            +7.62796053615234516538e+01,
            +1.68089224934630576269e+02,
            +1.59756391350164413639e+02,
            +7.05360906840444183113e+01,
            +1.40264691163389668864e+01,
            +9.99999999999999995305e-01,
    };

    static const T AD[] = {
            +5.67594532638770212846e-01,
            +1.47562562584847203173e+01,
            +8.45138970141474626562e+01,
            +1.77318088145400459522e+02,
            +1.64234692871529701831e+02,
            +7.14778400825575695274e+01,
            +1.40959135607834029598e+01,
            +1.00000000000000000470e+00,
    };

    static const T AFN[] = {
            -1.31696323418331795333e-01,
            -6.26456544431912369773e-01,
            -6.93158036036933542233e-01,
            -2.79779981545119124951e-01,
            -4.91900132609500318020e-02,
            -4.06265923594885404393e-03,
            -1.59276496239262096340e-04,
            -2.77649108155232920844e-06,
            -1.67787698489114633780e-08,
    };

    static const T AFD[] = {
            +1.33560420706553243746e+01,
            +3.26825032795224613948e+01,
            +2.67367040941499554804e+01,
            +9.18707402907259625840e+00,
            +1.47529146771666414581e+00,
            +1.15687173795188044134e-01,
            +4.40291641615211203805e-03,
            +7.54720348287414296618e-05,
            +4.51850092970580378464e-07,
    };

    static const T AGN[] = {
            +1.97339932091685679179e-02,
            +3.91103029615688277255e-01,
            +1.06579897599595591108e+00,
            +9.39169229816650230044e-01,
            +3.51465656105547619242e-01,
            +6.33888919628925490927e-02,
            +5.85804113048388458567e-03,
            +2.82851600836737019778e-04,
            +6.98793669997260967291e-06,
            +8.11789239554389293311e-08,
            +3.41551784765923618484e-10,
    };

    static const T AGD[] = {
            +9.30892908077441974853e+00,
            +1.98352928718312140417e+01,
            +1.55646628932864612953e+01,
            +5.47686069422975497931e+00,
            +9.54293611618961883998e-01,
            +8.64580826352392193095e-02,
            +4.12656523824222607191e-03,
            +1.01259085116509135510e-04,
            +1.17166733214413521882e-06,
            +4.91834570062930015649e-09,
    };

    int domain_flag = 0;

    T ai;

    if (std::isinf(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x > T(103.892)) {
        return T(0.0);
    }

    T f;
    T g;
    T k;

    if (x < T(-2.09)) {
        T z = T(1.0) / (T(-2.0) * x * std::sqrt(-x) / T(3.0));

        T afn = 0.0;

        for (uint8_t index = 0; index <= 8; index++) {
            afn = afn * (z * z) + AFN[index];
        }

        T afd = 0.0;

        for (uint8_t index = 0; index <= 8; index++) {
            afd = afd * (z * z) + AFD[index];
        }

        T agn = 0.0;

        for (uint8_t index = 0; index <= 10 + 0; index++) {
            agn = agn * (z * z) + AGN[index];
        }

        T agd = 0.0;

        for (uint8_t index = 0; index <= 10 - 1; index++) {
            agd = agd * (z * z) + AGD[index];
        }

        T t = T(-2.0) * x * std::sqrt(-x) / T(3.0) + T(0.25) * M_PI;

        return T(5.64189583547756286948e-01) / std::sqrt(std::sqrt(-x)) * (std::sin(t) * (T(1.0) + z * z * afn / afd) - std::cos(t) * (z * agn / agd));
    }

    if (x >= T(2.09)) {
        domain_flag = 5;

        T zeta = T(2.0) * x * std::sqrt(x) / T(3.0);

        T an = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            an = an * (T(1.0) / zeta) + AN[index];
        }

        T ad = 0.0;

        for (uint8_t index = 0; index <= 7; index++) {
            ad = ad * (T(1.0) / zeta) + AD[index];
        }

        ai = T(5.64189583547756286948e-01) * (an / ad) / (T(2.0) * std::sqrt(std::sqrt(x)) * std::exp(zeta));

        if (x > T(8.3203353)) {
            return ai;
        }
    }

    f = 1.0;
    g = x;
    k = 1.0;

    T m = 1.0;
    T n = x;
    T t = 1.0;
    T z = x * x * x;

    while (t > std::numeric_limits<T>::epsilon()) {
        m *= z;
        k += T(1.0);
        m /= k;
        n *= z;
        k += T(1.0);
        n /= k;
        m /= k;
        f += m;
        k += T(1.0);
        n /= k;
        g += n;

        t = std::abs(m / f);
    }

    if ((domain_flag & 1) == 0) {
        return T(0.355028053887817239260) * f - T(0.258819403792806798405) * g;
    }

    return ai;
} // T airy_ai(T x)
} // namespace special_functions
} // namespace native
} // namespace at
