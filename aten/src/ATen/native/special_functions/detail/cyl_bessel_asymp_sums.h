#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/numeric_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tnu, typename Tp>
struct cyl_bessel_asymp_sums_t {
  // FIXME: This will promote float to double if Tnu is integral.
  using Val = promote_t<Tnu, Tp>;
  Val Psum;
  Val Qsum;
  Val Rsum;
  Val Ssum;
};

template<typename T1, typename T2>
constexpr cyl_bessel_asymp_sums_t<T1, T2>
cyl_bessel_asymp_sums(T1 nu, T2 x, int sgn) {
  using T3 = promote_t<T1, T2>;
  using T4 = numeric_t<T3>;

  const auto nu_max = std::abs(T4(100) * (nu + T1(1)));
  auto k = 0;
  auto bk_xk = T3(1);
  auto Rsum = bk_xk;
  auto ak_xk = T3(1);
  auto Psum = ak_xk;
  auto convP = false;
  ++k;
  auto _2km1 = 1;
  bk_xk *= (T4(2) * nu * (T4(2) * nu) + 3) * (T2(1) / (T4(8) * x));
  auto Ssum = bk_xk;
  ak_xk *= (T4(2) * nu - _2km1) * (T4(2) * nu + _2km1) * (T2(1) / (T4(8) * x));
  auto Qsum = ak_xk;
  auto convQ = false;
  auto ak_xk_prev = std::abs(ak_xk);
  do {
    ++k;
    auto rk8x = T2(1) / (T4(8) * x) / T4(k);
    _2km1 += 2;
    bk_xk = sgn * (T4(2) * nu * (T4(2) * nu) + _2km1 * (_2km1 + 2)) * ak_xk * rk8x;
    Rsum += bk_xk;
    ak_xk *= sgn * (T4(2) * nu - _2km1) * (T4(2) * nu + _2km1) * rk8x;
    if (k > std::real(nu / T4(2)) && std::abs(ak_xk) > ak_xk_prev)
      break;
    Psum += ak_xk;
    ak_xk_prev = std::abs(ak_xk);
    convP = std::abs(ak_xk) < std::numeric_limits<T4>::epsilon() * std::abs(Psum);

    ++k;
    rk8x = T2(1) / (T4(8) * x) / T4(k);
    _2km1 += 2;
    bk_xk = (T4(2) * nu * (T4(2) * nu) + _2km1 * (_2km1 + 2)) * ak_xk * rk8x;
    Ssum += bk_xk;
    ak_xk *= (T4(2) * nu - _2km1) * (T4(2) * nu + _2km1) * rk8x;
    if (k > std::real(nu / T4(2)) && std::abs(ak_xk) > ak_xk_prev)
      break;
    Qsum += ak_xk;
    ak_xk_prev = std::abs(ak_xk);
    convQ = std::abs(ak_xk) < std::numeric_limits<T4>::epsilon() * std::abs(Qsum);

    if (convP && convQ)
      break;
  } while (k < nu_max);

  return {Psum, Qsum, Rsum, Ssum};
}
}
}
}
}

