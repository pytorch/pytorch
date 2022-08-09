#pragma once

namespace at::native::special_functions::detail {
template<typename T1>
T1
tricomi_confluent_hypergeometric_u(T1 a, T1 c, T1 x) {
  auto u = T1{};
  auto v = T1{};

  if (!is_integer(a) || is_integer(a)() > 0) {
    v = std::tgamma(c - T1(1)) * std::pow(x, T1(1) - c)
        * kummer_confluent_hypergeometric_1_f_1(a - c + T1(1), T1(2) - c, x) / std::tgamma(a);
  }

  if (!is_integer(a - c + T1(1)) || is_integer(a - c + T1(1))() > 0) {
    u = std::tgamma(T1(1) - c) * kummer_confluent_hypergeometric_1_f_1(a, c, x) / std::tgamma(a - c + T1(1));
  }

  return u + v;
}
}
