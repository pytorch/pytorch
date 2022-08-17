#pragma once

#include <vector>

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {

template<typename Tp>
Tp
stirling_2_series(unsigned int n, unsigned int m) {
  if (m > c10::numbers::factorials_size<Tp>()) {
    auto S2 = Tp{0};
    for (auto k = 0u; k <= m; ++k) {
      auto lf1 = ln_factorial<Tp>(k);
      auto lf2 = ln_factorial<Tp>(m - k);
      S2 += ((m - k) & 1 ? Tp{-1} : Tp{1})
          * std::exp(n * std::log(k) - lf1 - lf2);
    }
    return S2;
  } else {
    auto S2 = Tp{0};
    for (auto k = 0u; k <= m; ++k) {
      S2 += ((m - k) & 1 ? Tp{-1} : Tp{1})
          * std::pow(k, n)
          / factorial<Tp>(k)
          / factorial<Tp>(m - k);
    }
    // @todo Only round if the sum is less than
    // the maximum representable integer.
    // Find or make a tool for this.
    return std::nearbyint(S2);
  }
}

template<typename Tp>
std::vector<Tp>
stirling_2_recur(unsigned int n) {
  if (n == 0)
    return std::vector<Tp>(1, Tp{1});
  else {
    std::vector<Tp> sigold(n + 1), signew(n + 1);
    sigold[0] = signew[0] = Tp{0};
    sigold[1] = Tp{1};
    if (n == 1)
      return sigold;
    for (auto in = 1u; in <= n; ++in) {
      signew[1] = sigold[1];
      for (auto im = 2u; im <= n; ++im)
        signew[im] = im * sigold[im] + sigold[im - 1];
      std::swap(sigold, signew);
    }
    return signew;
  }
}

template<typename Tp>
std::vector<Tp>
stirling_2(unsigned int n) { return stirling_2_recur<Tp>(n); }

template<typename Tp>
Tp
log_stirling_2(unsigned int n, unsigned int m) {
  if (m > n)
    return -std::numeric_limits<Tp>::infinity();
  else if (m == n)
    return Tp{0};
  else if (m == 0 && n >= 1)
    return -std::numeric_limits<Tp>::infinity();
  else
    return std::log(stirling_number_2<Tp>(n, m));
}

template<typename Tp>
std::vector<Tp>
stirling_1_recur(unsigned int n) {
  if (n == 0)
    return std::vector<Tp>(1, Tp{1});
  else {
    std::vector<Tp> Sold(n + 1), Snew(n + 1);
    Sold[0] = Snew[0] = Tp{0};
    Sold[1] = Tp{1};
    if (n == 1)
      return Sold;
    for (auto in = 1u; in <= n; ++in) {
      for (auto im = 1u; im <= n; ++im)
        Snew[im] = Sold[im - 1] - in * Sold[im];
      std::swap(Sold, Snew);
    }
    return Snew;
  }
}

template<typename Tp>
std::vector<Tp>
stirling_1(unsigned int n) { return stirling_1_recur<Tp>(n); }

template<typename Tp>
Tp
log_stirling_1(unsigned int n, unsigned int m) {
  if (m > n)
    return -std::numeric_limits<Tp>::infinity();
  else if (m == n)
    return Tp{0};
  else if (m == 0 && n >= 1)
    return -std::numeric_limits<Tp>::infinity();
  else
    return std::log(std::abs(stirling_number_1<Tp>(n, m)));
}

template<typename Tp>
inline Tp
log_stirling_1_sign(unsigned int n, unsigned int m) { return (n + m) & 1 ? Tp{-1} : Tp{+1}; }

template<typename Tp>
inline constexpr std::vector<Tp>
bell_polynomial_b(unsigned int n) { return bell_series<Tp>(n); }

template<typename Tp, typename Up>
inline Up
bell(unsigned int n, Up x) {
  const auto Sn = stirling_2<Tp>(n);
  auto bell = Sn[n];
  for (unsigned int i = 1; i < n; ++i)
    bell = Sn[n - i] + x * bell;
  return bell;
}
}
}
}
}
