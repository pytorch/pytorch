#pragma once

#include <c10/util/numbers.h>
#include <ATen/native/special/detail/bernoulli_number.h>
#include <ATen/native/special/detail/is_integer.h>
#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/ln_gamma_sign.h>
#include <ATen/native/special/detail/is_complex_v.h>
#include <ATen/native/special/detail/ln_gamma.h>
#include <ATen/native/special/detail/lanczos_gamma_approximation.h>
#include <ATen/native/special/detail/gamma_reciprocal_series.h>

namespace at {
namespace native {
namespace special {
namespace detail {
// Predeclaration.
template<typename Tp>
Tp ln_gamma(Tp z);

template<typename T1>
T1
gamma(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto is_integer_z = is_integer(z);

  if (is_integer_z) {
    if (is_integer_z() <= 0) {
      return std::numeric_limits<T3>::quiet_NaN();
    } else if (is_integer_z() < static_cast<int>(c10::numbers::factorials_size<T3>())) {
      return static_cast<T3>(c10::numbers::factorials_v[is_integer_z() - 1]);
    } else {
      return std::numeric_limits<T3>::infinity();
    }
  } else if (std::real(z) > T3(1) && std::abs(z) < c10::numbers::factorials_size<T1>()) {
    auto p = T1(1);
    auto q = z;

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
    return at::native::special::ln_gamma_sign(z) * std::exp(ln_gamma(z));
  }
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
