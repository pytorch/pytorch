#pragma once

#include <ATen/native/special/cos_pi.h>
#include <ATen/native/special/detail/cyl_bessel_asymp_sums.h>
#include <ATen/native/special/detail/gamma.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/polar_pi.h>
#include <ATen/native/special/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
struct gamma_temme_t {
  T1 m;

  T1 positive;
  T1 negative;

  T1 gamma_1;
  T1 gamma_2;
};

template<typename T1>
gamma_temme_t<T1>
gamma_temme(T1 m) {
  using T2 = gamma_temme_t<T1>;

  if (std::abs(m) < std::numeric_limits<T1>::epsilon()) {
    return {m, T1(1), T1(1), -c10::numbers::egamma_v<T1>, T1(1)};
  } else if (std::real(m) <= T1(0)) {
    return {m, T1(+1) * gamma_reciprocal_series(T1(+1) + m), T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m, (T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m - T1(+1) * gamma_reciprocal_series(T1(+1) + m)) / (T1(2) * m), (T1(-1) * gamma_reciprocal_series(T1(-1) * m) / m + T1(+1) * gamma_reciprocal_series(T1(+1) + m)) / T1(2)};
  } else {
    return {m, T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m, T1(+1) * gamma_reciprocal_series(T1(+1) - m), (T1(+1) * gamma_reciprocal_series(T1(+1) - m) - T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m) / (T1(2) * m), (T1(+1) * gamma_reciprocal_series(T1(+1) - m) + T1(+1) * gamma_reciprocal_series(T1(+1) * m) / m) / T1(2)};
  }
}
}
}
}
}
