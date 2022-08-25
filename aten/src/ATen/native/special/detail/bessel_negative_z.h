#pragma once

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
bessel_t<T1, T1, c10::complex<T1>>
bessel_negative_z(T1 v, T1 z) {
  using T2 = c10::complex<T1>;
  using T3 = bessel_t<T1, T1, T2>;

  if (z >= T1(0)) {
    throw std::domain_error("non-negative `z`");
  } else {
    const auto positive_polar_pi_v = at::native::special::polar_pi(T1(1), +v);
    const auto negative_polar_pi_v = at::native::special::polar_pi(T1(1), -v);

    const auto cos_pi_v = at::native::special::cos_pi(v);

    const auto p = bessel(v, -z);
    const auto q = T2(0, 1) * T1(2);

    const auto j = positive_polar_pi_v * p.j;
    const auto y = negative_polar_pi_v * p.y + q * cos_pi_v * p.j;

    const auto j_derivative = -positive_polar_pi_v * p.j_derivative;
    const auto y_derivative = -negative_polar_pi_v * p.y_derivative - q * cos_pi_v * p.j_derivative;

    return {
        v,
        z,
        j,
        j_derivative,
        y,
        y_derivative,
    };
  }
}
}
}
}
}
