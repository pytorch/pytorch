#pragma once

#include <ATen/native/special_functions/detail/carlson_elliptic_r_c.h>
#include <ATen/native/special_functions/detail/carlson_elliptic_r_d.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
T1
carlson_elliptic_r_j(T1 x, T1 y, T1 z, T1 p) {
  using T2 = numeric_t<T1>;

  bool neg_arg = false;

  const auto is_complex_t1 = !is_complex_v < T1 >;

  if (is_complex_t1)
    if (std::real(x) < T2(0) || std::real(y) < T2(0) || std::real(z) < T2(0))
      neg_arg = true;

  if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isnan(p)) {
    return std::numeric_limits<T2>::quiet_NaN();
  } else if (neg_arg) {
    throw std::domain_error("carlson_elliptic_r_j: argument less than zero");
  } else {

    if (std::abs(x) + std::abs(y) < T2(5) * std::numeric_limits<T2>::min() || std::abs(y)
        + std::abs(z) < T2(5) * std::numeric_limits<T2>::min()
        || std::abs(z) + std::abs(x) < T2(5) * std::numeric_limits<T2>::min()
        || std::abs(p) < T2(5) * std::numeric_limits<T2>::min()) {
      throw std::domain_error("carlson_elliptic_r_j: argument too small");
    } else if (std::abs(p - z) < std::numeric_limits<T2>::epsilon()) {
      return carlson_elliptic_r_d(x, y, z);
    } else {
      auto xt = x;
      auto yt = y;
      auto zt = z;
      auto pt = p;
      auto A0 = (x + y + z + T2(2) * p) / T2(5);
      auto delta = (p - x) * (p - y) * (p - z);
      auto Q = std::pow(std::numeric_limits<T2>::epsilon() / T2(4), -T2(1) / T2(6))
          * std::max(std::abs(A0 - z), std::max(std::abs(A0 - x), std::max(std::abs(A0 - y), std::abs(A0 - p))));
      auto A = A0;
      auto f = T2(1);
      auto fe = T2(1);
      auto sum = T1(0);

      while (true) {
        auto xroot = std::sqrt(xt);
        auto yroot = std::sqrt(yt);
        auto zroot = std::sqrt(zt);
        auto proot = std::sqrt(pt);
        auto lambda = xroot * yroot + yroot * zroot + zroot * xroot;
        A = (A + lambda) / T2(4);
        xt = (xt + lambda) / T2(4);
        yt = (yt + lambda) / T2(4);
        zt = (zt + lambda) / T2(4);
        pt = (pt + lambda) / T2(4);
        auto d = (proot + xroot) * (proot + yroot) * (proot + zroot);
        auto E = delta / (fe * d * d);
        sum += carlson_elliptic_r_c(T1(1), T1(1) + E) / (f * d);
        f *= T2(4);
        fe *= T2(64);

        if (Q < f * std::abs(A)) {
          return (T2(1) - T2(3) * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
              + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
              * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)))) / T2(14)
              + ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) * ((A0 - z) / (f * A)) + T2(2)
                  * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                      + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)) + T1(4)
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)))) / T2(6) + T2(9)
              * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                  + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))
              * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                  + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)))) / T2(88) - T2(3)
              * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                  * (T2(2) * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) * ((A0 - z) / (f * A)))
                      + ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                          + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                              * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)) + T2(3)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))) / T2(22)
              - T2(9) * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                  + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))
                  * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) * ((A0 - z) / (f * A)) + T2(2)
                      * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) + (A0 - y) / (f * A) * ((A0 - z) / (f * A))
                          + (A0 - z) / (f * A) * ((A0 - x) / (f * A)) - T2(3)
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                              * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))))
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)) + T1(4)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2))
                          * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)))) / T2(52) + T2(3)
              * ((A0 - x) / (f * A) * ((A0 - y) / (f * A)) * ((A0 - z) / (f * A))
                  * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)
                      * (-((A0 - x) / (f * A) + (A0 - y) / (f * A) + (A0 - z) / (f * A)) / T2(2)))) / T2(26)) / f / A
              / std::sqrt(A) + T2(6) * sum;
        }
      }
    }
  }
}
}
}
}
}
