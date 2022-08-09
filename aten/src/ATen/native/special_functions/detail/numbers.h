#pragma once
#define MMATH_NUMBERS.h

namespace at::native::special_functions::detail {
template<typename T1>
inline constexpr T1
    inv_tau_v = T1{0.159154943091895335768883763372514362035L};

template<typename T1>
inline constexpr T1
    lnpi_v = T1{1.144729885849400174143427351353058711646L};

template<typename T1>
inline constexpr T1
    pi_sqr_div_6_v = T1{1.644934066848226436472415166646025189221L};

template<typename T1>
inline constexpr T1
    tau_v = T1{6.283185307179586476925286766559005768391L};

template<typename T1>
inline constexpr T1
    sqrtpi_v = T1{1.772453850905516027298167483341145182798L};

template<typename T1>
inline constexpr T1
    sqrttau_v = T1{2.506628274631000502415765284811045253010L};

inline constexpr double inv_tau = inv_tau_v<double>;
inline constexpr double lnpi = lnpi_v<double>;
inline constexpr double pi_sqr_div_6 = pi_sqr_div_6_v<double>;
inline constexpr double sqrtpi = sqrttau_v<double>;
inline constexpr double sqrttau = sqrttau_v<double>;
inline constexpr double tau = tau_v<double>;
}
