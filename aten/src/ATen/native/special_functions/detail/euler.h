#pragma once

#include <vector>

#include <emsr/sf_gamma.h> // binomial

namespace mmath {
namespace detail {

template<typename Tp>
Tp
euler_series(unsigned int n) {
  static constexpr std::size_t s_len = 22;
  static constexpr Tp
      s_num[s_len]
      {
          1ll, 0,
          -1ll, 0ll,
          5ll, 0ll,
          -61ll, 0ll,
          1385ll, 0ll,
          -50521ll, 0ll,
          2702765ll, 0ll,
          -199360981ll, 0ll,
          19391512145ll, 0ll,
          -2404879675441ll, 0ll,
          370371188237525ll, 0ll,
          //-69348874393137901ll, 0ll,
      };

  if (n == 0)
    return Tp{1};
  else if (n & 1)
    return Tp{0};
  else if (n == 2)
    return Tp{-1};
  else if (n < s_len)
    return s_num[n];
  else {
    std::vector<Tp> En(n + 1);
    En[0] = Tp{1};
    En[1] = Tp{0};
    En[2] = Tp{-1};

    for (auto i = 3u; i <= n; ++i) {
      En[i] = 0;

      if (i % 2 == 0) {
        for (auto j = 2u; j <= i; j += 2u)
          En[i] -= binomial<Tp>(i, j)
              * En[i - j];
      }
    }
    return En[n];
  }
}

template<typename Tp>
inline Tp
euler(unsigned int n) { return euler_series<Tp>(n); }

template<typename Tp>
Tp
euler(unsigned int n, Tp x) {
  if (std::isnan(x))
    return std::numeric_limits<Tp>::quiet_NaN();
  else {
    auto bx1 = bernoulli(n + 1, x);
    auto bx2 = bernoulli(n + 1, Tp{0.5L} * x);

    auto E_n = Tp{2} * (bx1 - bx2 * std::pow(Tp{2}, Tp(n + 1)))
        / Tp(n + 1);

    return E_n;
  }
}

template<typename Tp>
Tp
eulerian_1_recur(unsigned int n, unsigned int m) {
  if (m == 0)
    return Tp{1};
  else if (m >= n)
    return Tp{0};
  else if (m == n - 1)
    return Tp{1};
  else if (n - m - 1 < m) // Symmetry.
    return eulerian_1_recur<Tp>(n, n - m - 1);
  else {
    // Start recursion with n == 2 (already returned above).
    std::vector<Tp> Aold(m + 1), Anew(m + 1);
    Aold[0] = Tp{1};
    Anew[0] = Tp{1};
    Anew[1] = Tp{1};
    for (auto in = 3u; in <= n; ++in) {
      std::swap(Aold, Anew);
      for (auto im = 1u; im <= m; ++im)
        Anew[im] = (in - im) * Aold[im - 1]
            + (im + 1) * Aold[im];
    }
    return Anew[m];
  }
}

template<typename Tp>
inline Tp
eulerian_1(unsigned int n, unsigned int m) { return eulerian_1_recur<Tp>(n, m); }

template<typename Tp>
std::vector<Tp>
eulerian_1_recur(unsigned int n) {
  if (n == 0)
    return std::vector<Tp>(1, Tp{1});
    //else if (m == n - 1)
    //return Tp{1};
    //else if (n - m - 1 < m) // Symmetry.
    //return eulerian_1_recur<Tp>(n, n - m - 1);
  else {
    // Start recursion with n == 2 (already returned above).
    std::vector<Tp> Aold(n + 1), Anew(n + 1);
    Aold[0] = Anew[0] = Tp{1};
    Anew[1] = Tp{1};
    for (auto in = 3u; in <= n; ++in) {
      std::swap(Aold, Anew);
      for (auto im = 1u; im <= n; ++im)
        Anew[im] = (in - im) * Aold[im - 1]
            + (im + 1) * Aold[im];
    }
    return Anew;
  }
}

template<typename Tp>
inline constexpr std::vector<Tp>
eulerian_1(unsigned int n) { return eulerian_1_recur<Tp>(n); }

template<typename Tp>
Tp
eulerian_2_recur(unsigned int n, unsigned int m) {
  if (m == 0)
    return Tp{1};
  else if (m >= n)
    return Tp{0};
  else if (n == 0)
    return Tp{1};
  else {
    // Start recursion with n == 2 (already returned above).
    std::vector<Tp> Aold(m + 1), Anew(m + 1);
    Aold[0] = Tp{1};
    Anew[0] = Tp{1};
    Anew[1] = Tp{2};
    for (auto in = 3u; in <= n; ++in) {
      std::swap(Aold, Anew);
      for (auto im = 1u; im <= m; ++im)
        Anew[im] = (2 * in - im - 1) * Aold[im - 1]
            + (im + 1) * Aold[im];
    }
    return Anew[m];
  }
}

template<typename Tp>
inline Tp
eulerian_2(unsigned int n, unsigned int m) { return eulerian_2_recur<Tp>(n, m); }

template<typename Tp>
std::vector<Tp>
eulerian_2_recur(unsigned int n) {
  if (n == 0)
    return std::vector<Tp>(1, Tp{1});
    //else if (m >= n)
    //return Tp{0};
    //else if (n == 0)
    //return Tp{1};
  else {
    // Start recursion with n == 2 (already returned above).
    std::vector<Tp> Aold(n + 1), Anew(n + 1);
    Aold[0] = Anew[0] = Tp{1};
    Anew[1] = Tp{2};
    for (auto in = 3u; in <= n; ++in) {
      std::swap(Aold, Anew);
      for (auto im = 1u; im <= n; ++im)
        Anew[im] = (2 * in - im - 1) * Aold[im - 1]
            + (im + 1) * Aold[im];
    }
    return Anew;
  }
}

template<typename Tp>
inline constexpr std::vector<Tp>
eulerian_2(unsigned int n) { return eulerian_2_recur<Tp>(n); }

} // namespace detail
} // namespace emsr
