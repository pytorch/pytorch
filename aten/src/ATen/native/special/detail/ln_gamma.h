#pragma once

#include <ATen/native/special/detail/is_complex_v.h>
#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/spouge_ln_gamma_approximation.h>
#include <ATen/native/special/sin_pi.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
ln_gamma(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::real(z) < T3(0.5L)) {
    if (std::abs(at::native::special::sin_pi(z)) < std::numeric_limits<T3>::epsilon()) {
      return std::numeric_limits<T3>::infinity();
    } else {
      return c10::numbers::lnpi_v<T1> - std::log(std::abs(at::native::special::sin_pi(z))) - ln_gamma(T2(1) - z);
    }
  } else if (std::real(z) > T3(1) && std::abs(z) < c10::numbers::factorials_size<T1>()) {
    auto p = z;
    auto q = T1(1);

    while (std::real(p) > T3(1)) {
      p = p - T3(1);

      q = p * q;
    }

    return std::log(q) + ln_gamma(p);
  } else {
    return spouge_ln_gamma_approximation(z - T3(1));
  }
}

template<typename T1>
c10::complex<T1>
ln_gamma(c10::complex<T1> z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  using T4 = c10::complex<T3>;

  const auto is_integer_a = is_integer(z);

  if (is_integer_a) {
    const auto integer_a = is_integer_a();

    if (integer_a <= 0) {
      return std::numeric_limits<T3>::quiet_NaN();
    } else if (integer_a < static_cast<int>(c10::numbers::factorials_size<T3>())) {
      return T3((c10::numbers::log_factorials_v[integer_a - 1]));
    } else {
      return ln_gamma(T3(integer_a));
    }
  } else if (std::real(z) >= T3(0.5L)) {
    return spouge_ln_gamma_approximation(z - T3(1));
  } else {
    if (std::abs(at::native::special::sin_pi(z)) < std::numeric_limits<T3>::epsilon()) {
      return T4(std::numeric_limits<T3>::quiet_NaN(), T3(0));
    } else {
      return c10::numbers::lnpi_v<T3> - std::log(at::native::special::sin_pi(z)) - ln_gamma(T2(1) - z);
    }
  }
}
}
}
}
}
