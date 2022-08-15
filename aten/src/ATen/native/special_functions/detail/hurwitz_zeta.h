#pragma once

#include <ATen/native/special_functions/detail/zeta.h>
#include <ATen/native/special_functions/detail/euler_maclaurin.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/riemann_zeta.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
hurwitz_zeta_euler_maclaurin(T1 s, T1 a) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto ans = std::pow(T2{10 + std::numeric_limits<T3>::digits10 / 2} + a, -s)
      * ((T2{10 + std::numeric_limits<T3>::digits10 / 2} + a) / (s - T2{1}) + T3{0.5L});

  for (auto k = 0; k < 10 + std::numeric_limits<T3>::digits10 / 2; ++k)
    ans += std::pow(T1(k) + a, -s);

  auto sfact = s;
  auto pfact = std::pow(T2{10 + std::numeric_limits<T3>::digits10 / 2} + a, -s)
      / (T2(10 + std::numeric_limits<T3>::digits10 / 2) + a);
  for (auto j = 0; j < EULER_MACLAURIN_SIZE - 1; ++j) {
    auto delta = T3(EULER_MACLAURIN[j + 1])
        * sfact * pfact;
    ans += delta;
    if (std::abs(delta) < T3{0.5L} * std::numeric_limits<T3>::epsilon() * std::abs(ans))
      break;
    sfact *= (s + T2(2 * j + 1)) * (s + T2(2 * j + 2));
    pfact /=
        (T2{10 + std::numeric_limits<T3>::digits10 / 2} + a) * (T2{10 + std::numeric_limits<T3>::digits10 / 2} + a);
  }

  return ans;
}

template<typename T1>
T1
hurwitz_zeta(T1 s, T1 a) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(s) || std::isnan(a)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (a == T3(1)) {
    if (s == T3(1)) {
      return std::numeric_limits<T3>::infinity();
    } else {
      return riemann_zeta(s);
    }
  } else {
    return hurwitz_zeta_euler_maclaurin(s, a);
  }
}
}
}
}
}
