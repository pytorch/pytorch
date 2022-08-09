#pragma once

namespace at::native::special_functions::detail {
template<typename Tp>
Tp
gamma_q(Tp a, Tp x) {
  using Val = Tp;
  using Real = numeric_t<Val>;

  if (std::isnan(a) || std::isnan(x))
    return std::numeric_limits<Tp>::quiet_NaN();

  if (is_integer(a) && is_integer(a)() <= 0)
    throw std::domain_error("gamma_q: non-positive integer argument a");
  else if (std::real(x) < std::real(a + Real{1}))
    return Val{1} - gamma_series(a, x).first;
  else
    return gamma_continued_fraction(a, x).first;
}
}
