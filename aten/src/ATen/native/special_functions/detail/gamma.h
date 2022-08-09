#pragma once

#include <c10/util/numbers.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/ln_gamma_sign.h>
#include <ATen/native/special_functions/detail/is_complex_v.h>
#include <ATen/native/special_functions/detail/ln_gamma.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
// Predeclaration.
template<typename Tp>
Tp ln_gamma(Tp x);

template<typename T1>
constexpr T1
log_gamma_bernoulli(T1 x) {
  using Val = T1;
  using Real = numeric_t<Val>;

  auto lg = (x - Real{0.5L}) * std::log(x)
      - x + Real{0.5L} * (c10::numbers::ln2_v<Real> + c10::numbers::lnpi_v<Real>);

  auto xk = Real{1} / x;
  for (unsigned int i = 1; i < 100; ++i) {
    lg += bernoulli_number<T1>(T1(2 * i)) * xk
        / (T1(2 * i) * (T1(2 * i) - T1(1)));
    if (std::abs(bernoulli_number<T1>(T1(2 * i)) * xk
                     / (T1(2 * i)
                         * (T1(2 * i) - T1(1)))) < Real{0.01L} * std::numeric_limits<Real>::epsilon() * std::abs(lg))
      break;
    xk *= Real{1} / (x * x);
  }

  return lg;
}

template<typename Tp>
struct gamma_lanczos_data {
};

template<>
struct gamma_lanczos_data<float> {
  static constexpr float s_g = 6.5F;
  static constexpr std::array<float, 7>
      s_cheby
      {
          3.307139e+02F,
          -2.255998e+02F,
          6.989520e+01F,
          -9.058929e+00F,
          4.110107e-01F,
          -4.150391e-03F,
          -3.417969e-03F,
      };
};

template<>
struct gamma_lanczos_data<double> {
  static constexpr double s_g = 9.5;
  static constexpr std::array<double, 10>
      s_cheby
      {
          5.557569219204146e+03,
          -4.248114953727554e+03,
          1.881719608233706e+03,
          -4.705537221412237e+02,
          6.325224688788239e+01,
          -4.206901076213398e+00,
          1.202512485324405e-01,
          -1.141081476816908e-03,
          2.055079676210880e-06,
          1.280568540096283e-09,
      };
};

template<>
struct gamma_lanczos_data<long double> {
  static constexpr long double s_g = 10.5L;
  static constexpr std::array<long double, 11>
      s_cheby
      {
          1.440399692024250728e+04L,
          -1.128006201837065341e+04L,
          5.384108670160999829e+03L,
          -1.536234184127325861e+03L,
          2.528551924697309561e+02L,
          -2.265389090278717887e+01L,
          1.006663776178612579e+00L,
          -1.900805731354182626e-02L,
          1.150508317664389324e-04L,
          -1.208915136885480024e-07L,
          -1.518856151960790157e-10L,
      };
};

template<typename Tp>
constexpr Tp
lanczos_binet1p(Tp z) {
  using Val = Tp;
  using Real = numeric_t<Val>;
  const auto s_sqrt_2pi = c10::numbers::sqrttau_v<Real>;
  const auto c = gamma_lanczos_data<Real>::s_cheby;
  //auto g =  gamma_lanczos_data<Real>::s_g;

  auto fact = Val{1};
  auto sum = Val{0.5L} * c[0];
  for (unsigned int k = 1, n = c.size(); k < n; ++k) {
    fact *= (z - Real(k - 1)) / (z + Real(k));
    sum += fact * c[k];
  }
  return s_sqrt_2pi * sum;
}

template<typename T1>
constexpr T1
lanczos_log_gamma1p(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;
  const auto s_ln_pi = c10::numbers::lnpi_v<T3>;
  auto g = gamma_lanczos_data<T3>::s_g;
  // Reflection for z < -1.
  if (std::real(z) < T3(-1)) {
    auto sin_fact = at::native::special_functions::sin_pi(z);
    if (!is_complex_v<T2>)
      sin_fact = std::abs(sin_fact);
    return s_ln_pi - std::log(sin_fact)
        - lanczos_log_gamma1p(-T3(1) - z);
  } else {
    auto sum = lanczos_binet1p(z);
    if (!is_complex_v<T2>)
      sum = std::abs(sum);
    return std::log(sum)
        + (z + T3{0.5L}) * std::log(z + g + T3{0.5L})
        - (z + g + T3{0.5L});
  }
}

template<typename T1>
T1
gamma_reciprocal_series(T1 x) {
  using T2 = numeric_t<T1>;

  static constexpr long double series[31] = {
      +0.0000000000000000000000000000000000000000L,
      +1.0000000000000000000000000000000000000000L,
      +0.5772156649015328606065120900824024310422L,
      -0.6558780715202538810770195151453904812798L,
      -0.0420026350340952355290039348754298187114L,
      +0.1665386113822914895017007951021052357178L,
      -0.0421977345555443367482083012891873913017L,
      -0.0096219715278769735621149216723481989754L,
      +0.0072189432466630995423950103404465727099L,
      -0.0011651675918590651121139710840183886668L,
      -0.0002152416741149509728157299630536478065L,
      +0.0001280502823881161861531986263281643234L,
      -0.0000201348547807882386556893914210218184L,
      -0.0000012504934821426706573453594738330922L,
      +0.0000011330272319816958823741296203307449L,
      -0.0000002056338416977607103450154130020573L,
      +0.0000000061160951044814158178624986828553L,
      +0.0000000050020076444692229300556650480600L,
      -0.0000000011812745704870201445881265654365L,
      +0.0000000001043426711691100510491540332312L,
      +0.0000000000077822634399050712540499373114L,
      -0.0000000000036968056186422057081878158781L,
      +0.0000000000005100370287454475979015481323L,
      -0.0000000000000205832605356650678322242954L,
      -0.0000000000000053481225394230179823700173L,
      +0.0000000000000012267786282382607901588938L,
      -0.0000000000000001181259301697458769513765L,
      +0.0000000000000000011866922547516003325798L,
      +0.0000000000000000014123806553180317815558L,
      -0.0000000000000000002298745684435370206592L,
      +0.0000000000000000000171440632192733743338L,
  };

  auto p = T1(1);
  auto q = T1(0);

  for (auto j = 1; j < 31; j++) {
    p = p * x;

    const auto r = T1(series[j]) * p;

    q = q + r;

    if (std::abs(r) < std::numeric_limits<T2>::epsilon()) {
      break;
    }
  }

  return q;
}

template<typename T1>
T1
gamma(T1 a) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (is_integer(a)) {
    if (is_integer(a)() <= 0) {
      return std::numeric_limits<T3>::quiet_NaN();
    } else if (is_integer(a)() < static_cast<int>(c10::numbers::factorials_size<T3>)) {
      return static_cast<T3>(c10::numbers::factorials_v[is_integer(a)() - 1]);
    } else {
      return std::numeric_limits<T3>::infinity();
    }
  } else if (std::real(a) > T3(1) && std::abs(a) < c10::numbers::factorials_size<T1>) {
    auto p = T1(1);
    auto q = a;

    while (std::real(q) > T3(1)) {
      q = q - T3(1);
      p = p * q;
    }

    static constexpr long double series[31] = {
        +0.0000000000000000000000000000000000000000L,
        +1.0000000000000000000000000000000000000000L,
        +0.5772156649015328606065120900824024310422L,
        -0.6558780715202538810770195151453904812798L,
        -0.0420026350340952355290039348754298187114L,
        +0.1665386113822914895017007951021052357178L,
        -0.0421977345555443367482083012891873913017L,
        -0.0096219715278769735621149216723481989754L,
        +0.0072189432466630995423950103404465727099L,
        -0.0011651675918590651121139710840183886668L,
        -0.0002152416741149509728157299630536478065L,
        +0.0001280502823881161861531986263281643234L,
        -0.0000201348547807882386556893914210218184L,
        -0.0000012504934821426706573453594738330922L,
        +0.0000011330272319816958823741296203307449L,
        -0.0000002056338416977607103450154130020573L,
        +0.0000000061160951044814158178624986828553L,
        +0.0000000050020076444692229300556650480600L,
        -0.0000000011812745704870201445881265654365L,
        +0.0000000001043426711691100510491540332312L,
        +0.0000000000077822634399050712540499373114L,
        -0.0000000000036968056186422057081878158781L,
        +0.0000000000005100370287454475979015481323L,
        -0.0000000000000205832605356650678322242954L,
        -0.0000000000000053481225394230179823700173L,
        +0.0000000000000012267786282382607901588938L,
        -0.0000000000000001181259301697458769513765L,
        +0.0000000000000000011866922547516003325798L,
        +0.0000000000000000014123806553180317815558L,
        -0.0000000000000000002298745684435370206592L,
        +0.0000000000000000000171440632192733743338L,
    };

    auto r = T1(1);
    auto s = T1(0);

    for (auto j = 1; j < 31; j++) {
      r = r * q;

      const auto t = T1(series[j]) * r;

      s = s + t;

      if (std::abs(t) < std::numeric_limits<T2>::epsilon()) {
        break;
      }
    }

    return p / s;
  } else {
    return log_gamma_sign(a) * std::exp(ln_gamma(a));
  }
}

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

    for (unsigned int j = 1; j <= 10 * int(10 + std::sqrt(std::abs(a))); j++) {
      p = p + T3(1);
      q = q * (x / p);
      r = r + q;

      if (std::abs(q) < T3(3) * std::numeric_limits<T1>::epsilon() * std::abs(r)) {
        auto gamser = std::exp(-x + a * std::log(x) - lngam) * r * log_gamma_sign(a);

        return std::make_pair(gamser, lngam);
      }
    }

    throw std::logic_error("");
  }
}

template<typename T1>
std::pair<T1, T1>
gamma_continued_fraction(T1 a, T1 x) {
  using T2 = T1;
  using T3 = at::native::special_functions::detail::numeric_t<T2>;

  auto lngam = at::native::special_functions::detail::ln_gamma(a);
  auto sign = at::native::special_functions::detail::log_gamma_sign(a);

  auto b = x + T3(1) - a;
  auto c = T3(1) / (T3(3) * std::numeric_limits<T1>::min());
  auto d = T3(1) / b;
  auto h = d;

  for (unsigned int j = 1; j <= 10 * int(10 + std::sqrt(std::abs(a))); ++j) {
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

    h *= del;

    if (std::abs(del - T3(1)) < T3(3) * std::numeric_limits<T1>::epsilon()) {
      auto gamcf = std::exp(-x + a * std::log(x) - lngam) * h * sign;
      return std::make_pair(gamcf, lngam);
    }
  }

  throw std::logic_error("");
}

template<typename T1>
std::pair<T1, T1>
gamma(T1 a, T1 x) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  if (std::isnan(a) || std::isnan(x)) {
    return std::make_pair(std::numeric_limits<T1>::quiet_NaN(), std::numeric_limits<T1>::quiet_NaN());
  } else if (is_integer(a) && is_integer(a)() <= 0) {
    throw std::domain_error("non-positive integer `a`");
  } else if (std::real(x) < std::real(a + T3(1))) {
    auto Pgam = gamma_series(a, x).first;
    return std::make_pair(Pgam, T2(1) - Pgam);
  } else {
    auto Qgam = gamma_continued_fraction(a, x).first;
    return std::make_pair(T2(1) - Qgam, Qgam);
  }
}
}
}
}
}
