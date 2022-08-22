#pragma once

#include <ATen/native/special/detail/complex.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/ln_gamma_sign.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
std::pair<T1, T1>
gamma_continued_fraction(T1 a, T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto b = x + T3(1) - a;
  auto c = T3(1) / (T3(3) * std::numeric_limits<T1>::min());
  auto d = T3(1) / b;
  auto h = d;

  for (unsigned int j = 1; j <= 10 * int(10 + std::sqrt(std::abs(a))); j++) {
    auto an = -T3(j) * (T3(j) - a);

    b = b + T3(2);
    d = an * d + b;

    if (std::abs(d) < T3(3) * std::numeric_limits<T1>::min()) {
      d = T3(3) * std::numeric_limits<T1>::min();
    }

    c = b + an / c;

    if (std::abs(c) < T3(3) * std::numeric_limits<T1>::min()) {
      c = T3(3) * std::numeric_limits<T1>::min();
    }

    d = T3(1) / d;

    auto del = d * c;

    h = h * del;

    if (std::abs(del - T3(1)) < T3(3) * std::numeric_limits<T1>::epsilon()) {
      return std::make_pair(std::exp(-x + a * std::log(x) - ln_gamma(a)) * h * at::native::special::ln_gamma_sign(a), ln_gamma(a));
    }
  }

  throw std::logic_error("");
}
}
}
}
}
