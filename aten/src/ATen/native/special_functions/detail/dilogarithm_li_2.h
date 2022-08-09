#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
dilogarithm_li_2(T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(x)) {
    return std::numeric_limits<T3>::quiet_NaN();
  } else if (x > T1{+1}) {
    throw std::range_error("dilogarithm_li_2: argument greater than one");
  } else if (x < T1(-1)) {
    return -dilogarithm_li_2(T1(1) - T1(1) / (T1(1) - x)) - T1{0.5L} * std::log(T1(1) - x) * std::log(T1(1) - x);
  } else if (x == T1(0)) {
    return T1(0);
  } else if (x == T1(1)) {
    return c10::numbers::pi_sqr_div_6_v<T3>;
  } else if (x == -T1(1)) {
    return -T1{0.5L} * c10::numbers::pi_sqr_div_6_v<T3>;
  } else if (x > T1{0.5L}) {
    return c10::numbers::pi_sqr_div_6_v<T3> - std::log(x) * std::log(T1(1) - x) - dilogarithm_li_2(T1(1) - x);
  } else if (x < -T1{0.5L}) {
    return -T1{0.5L} * c10::numbers::pi_sqr_div_6_v<T3> - std::log(T1(1) - x) * std::log(-x)
        + dilogarithm_li_2(T1(1) + x) - T1{0.5L} * dilogarithm_li_2(T1(1) - x * x);
  } else {
    T1 sum = 0;
    T1 fact = 1;

    for (auto j = 1ULL; j < 100000ULL; j++) {
      fact *= x;
      auto term = fact / (j * j);
      sum += term;

      if (std::abs(term) < 10 * std::numeric_limits<T3>::epsilon() * std::abs(sum)) { break; }
      if (j + 1 == 100000ULL) { throw std::runtime_error("dilogarithm_li_2: sum failed"); }
    }

    return sum;
  }
}
}
