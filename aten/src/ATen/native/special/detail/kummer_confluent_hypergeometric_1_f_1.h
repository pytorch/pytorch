#pragma once

#include <ATen/native/special/detail/numeric_t.h>
#include <ATen/native/special/detail/is_integer.h>

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
T1
kummer_confluent_hypergeometric_1_f_1(T1 a, T1 c, T1 z) {
  using T2 = numeric_t<T1>;

  if (std::isnan(a) || std::isnan(c) || std::isnan(z)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    const auto is_integer_c = is_integer(c);
    
    if (is_integer_c && is_integer_c() <= 0) {
      return std::numeric_limits<T1>::infinity();
    } else if (a == T1(0)) {
      return T1(1);
    } else if (c == a) {
      return std::exp(z);
    } else if (z < T1(0)) {
      auto F = T1(1);

      auto qc = T1(1);

      const auto a_1 = a + T2(1);
      const auto c_2 = c * T2(2);
      const auto a_2 = a + T2(2);
      const auto c_1 = c + T2(1);

      const auto bbf = c_1 * T2(2);
      const auto bbg = a_1 / c_2;

      const auto po = bbg * -z;
      
      auto qb = T1(1) + po;

      const auto pm = a_2 / bbf;
      const auto pq = bbg / T2(3);
      const auto pr = pq * -z;
      const auto pp = T2(1) + pr;
      const auto ps = pm * -z * pp;

      auto qa = T1(1) + ps;

      auto pc = T1(1);
      const auto pe = a / c;
      auto pb = qb - pe * -z;

      const auto pd = T1(1) + pm * -z;

      const auto pf = pe * pd;
      const auto pg = pf * -z;
      const auto ph = pe * bbg;
      const auto pi = c / c_1;
      const auto pj = ph * pi;
      const auto pk = pj * -z;
      const auto pl = pk * -z;

      auto pa = qa - pg + pl;

      int n = 3;

      while (true) {
        const auto f = n * T2(2);
        const auto i = n - T2(1);
        const auto k = n - T2(2);
        const auto g = f - T2(3);
        const auto j = g * T2(2);

        const auto d = i + a;
        const auto l = k - a;
        const auto e = i + c;
        const auto h = k + c;
        const auto m = -z * (l / (j * e)) + T2(1);
        const auto o = -z * (-z * ((a + T2(n)) * (d / (e * ((f - T2(1)) * T2(4) * (g * h))))) + (-d * ((i - c) / (e * (j * h)))));
        const auto p = z * (z * (z * (d * ((k + a) * (l / (e * (h * ((f - T2(5)) * (g * T2(8) * (g * (n - T2(3) + c)))))))))));

        auto pn = pa * m + ((pb * o) + (pc * p));
        auto qn = qa * m + ((qb * o) + (qc * p));

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
        p = p * ((a + T1(j)) * z / ((c + T1(j)) * T1(1 + j)));
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
}
}
}
}
