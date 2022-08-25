#pragma once

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
std::pair<T1, T1>
gamma_series(T1 a, T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto lngam = ln_gamma(a);

  if (is_integer(a) && is_integer(a)() <= 0) {
    throw std::domain_error("non-positive integer `a`");
  } else if (x == T3(0)) {
    return std::make_pair(T2(0), lngam);
  } else if (std::real(x) < T3(0)) {
    throw std::domain_error("negative `x`");
  } else {
    auto p = a;
    T2 q;
    T2 r;

    q = r = T1(1) / a;

    for (auto j = 1; j <= 10 * int(10 + std::sqrt(std::abs(a))); j++) {
      p = p + T3(1);
      q = q * (x / p);
      r = r + q;

      if (std::abs(q) < T3(3) * std::numeric_limits<T1>::epsilon() * std::abs(r)) {
        auto gamser = std::exp(-x + a * std::log(x) - lngam) * r * at::native::special::ln_gamma_sign(a);

        return std::make_pair(gamser, lngam);
      }
    }

    throw std::logic_error("");
  }
}
}
}
}
}
