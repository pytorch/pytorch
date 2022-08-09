#pragma once

#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/is_integer.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
kummer_confluent_hypergeometric_1_f_1(T1 a, T1 c, T1 x) {
  using T2 = numeric_t<T1>;

  if (std::isnan(a) || std::isnan(c) || std::isnan(x)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (is_integer(c) && is_integer(c)() <= 0) {
    return std::numeric_limits<T1>::infinity();
  } else if (a == T1(0)) {
    return T1(1);
  } else if (c == a) {
    return std::exp(x);
  } else if (x < T1(0)) {
    auto F = T1(1);

    auto qc = T1(1);
    auto qb = T1(1) + (a + T2(1)) / (T2(2) * c) * -x;
    auto qa = T1(1) + (a + T2(2)) / (T2(2) * (c + T2(1))) * -x * (T2(1) + (a + T2(1)) / (T2(2) * c) / T2(3) * -x);

    auto pc = T1(1);
    auto pb = qb - a / c * -x;
    auto pa = qa - a / c * (T1(1) + (a + T2(2)) / (T2(2) * (c + T2(1))) * -x) * -x
        + a / c * ((a + T2(1)) / (T2(2) * c)) * (c / (c + T2(1))) * -x * -x;

    int n = 3;

    while (true) {
      auto pn = (T2(1) + (T2(n - 2) - a) / (T2(2 * T2(2 * n - 3)) * (T2(n - 1) + c)) * -x) * pa
          + (-(T2(n - 1) + a) * (T2(n - 1) - c) / (T2(2 * T2(2 * n - 3)) * (T2(n - 2) + c) * (T2(n - 1) + c))
              + (T2(n) + a) * (T2(n - 1) + a)
                  / (T2(4 * T2(2 * n - 1) * T2(2 * n - 3)) * (T2(n - 2) + c) * (T2(n - 1) + c)) * -x) * -x * pb
          + -(T2(n - 2) + a) * (T2(n - 1) + a) * (T2(n - 2) - a)
              / (T2(8 * T2(2 * n - 3) * T2(2 * n - 3) * T2(2 * n - 5)) * (T2(n - 3) + c) * (T2(n - 2) + c)
                  * (T2(n - 1) + c)) * (-x * -x * -x) * pc;
      auto qn = (T2(1) + (T2(n - 2) - a) / (T2(2 * T2(2 * n - 3)) * (T2(n - 1) + c)) * -x) * qa
          + (-(T2(n - 1) + a) * (T2(n - 1) - c) / (T2(2 * T2(2 * n - 3)) * (T2(n - 2) + c) * (T2(n - 1) + c))
              + (T2(n) + a) * (T2(n - 1) + a)
                  / (T2(4 * T2(2 * n - 1) * T2(2 * n - 3)) * (T2(n - 2) + c) * (T2(n - 1) + c)) * -x) * -x * qb
          + -(T2(n - 2) + a) * (T2(n - 1) + a) * (T2(n - 2) - a)
              / (T2(8 * T2(2 * n - 3) * T2(2 * n - 3) * T2(2 * n - 5)) * (T2(n - 3) + c) * (T2(n - 2) + c)
                  * (T2(n - 1) + c)) * (-x * -x * -x) * qc;
      auto rn = pn / qn;

      const auto prec = std::abs((F - rn) / F);

      F = rn;

      if (prec < std::numeric_limits<T2>::epsilon() || n > 20000) {
        break;
      }

      const auto abs_qn = std::abs(qn);
      const auto abs_pn = std::abs(pn);

      const auto t = std::pow(std::numeric_limits<T1>::max(), 1 / T2(6));

      if (abs_pn > t || abs_qn > t) {
        pn /= t;
        qn /= t;
        pa /= t;
        qa /= t;
        pb /= t;
        qb /= t;
        pc /= t;
        qc /= t;
      } else if (abs_pn < T1(1) / t || abs_qn < T1(1) / t) {
        pn *= t;
        qn *= t;
        pa *= t;
        qa *= t;
        pb *= t;
        qb *= t;
        pc *= t;
        qc *= t;
      }

      n++;

      qc = qb;
      qb = qa;
      qa = qn;
      pc = pb;
      pb = pa;
      pa = pn;
    }

    if (n >= 20000) {
      throw std::runtime_error("iteration failed to converge");
    }

    return F;
  } else {
    auto p = T1(1);
    auto q = T1(1);

    unsigned int j;

    for (j = 0; j < 100000; j++) {
      p = p * ((a + T1(j)) * x / ((c + T1(j)) * T1(1 + j)));
      q = q + p;

      if (std::abs(p) < std::numeric_limits<T1>::epsilon()) {
        break;
      }
    }

    if (j == 100000) {
      throw std::runtime_error("series failed to converge");
    }

    return q;
  }
}
}
