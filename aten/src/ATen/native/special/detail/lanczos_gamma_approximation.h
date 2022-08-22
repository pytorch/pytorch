#pragma once

namespace at {
namespace native {
namespace special {
namespace detail {
template<typename T1>
struct lanczos_gamma_approximation;

template<>
struct lanczos_gamma_approximation<float> {
  static constexpr float s_g = 6.5F;
  static constexpr std::array<float, 7> coefficients = {
      +3.307139e+02F,
      -2.255998e+02F,
      +6.989520e+01F,
      -9.058929e+00F,
      +4.110107e-01F,
      -4.150391e-03F,
      -3.417969e-03F,
  };
};

template<>
struct lanczos_gamma_approximation<double> {
  static constexpr double s_g = 9.5;
  static constexpr std::array<double, 10> coefficients = {
      +5.557569219204146e+03,
      -4.248114953727554e+03,
      +1.881719608233706e+03,
      -4.705537221412237e+02,
      +6.325224688788239e+01,
      -4.206901076213398e+00,
      +1.202512485324405e-01,
      -1.141081476816908e-03,
      +2.055079676210880e-06,
      +1.280568540096283e-09,
  };
};

template<>
struct lanczos_gamma_approximation<long double> {
  static constexpr long double s_g = 10.5L;
  static constexpr std::array<long double, 11> coefficients = {
      +1.440399692024250728e+04L,
      -1.128006201837065341e+04L,
      +5.384108670160999829e+03L,
      -1.536234184127325861e+03L,
      +2.528551924697309561e+02L,
      -2.265389090278717887e+01L,
      +1.006663776178612579e+00L,
      -1.900805731354182626e-02L,
      +1.150508317664389324e-04L,
      -1.208915136885480024e-07L,
      -1.518856151960790157e-10L,
  };
};

template<typename T1>
T1
lanczos_binet1p(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto p = T2(1);
  auto q = T2(0.5L) * lanczos_gamma_approximation<T3>::coefficients[0];

  for (unsigned int k = 1, n = lanczos_gamma_approximation<T3>::coefficients.size(); k < n; ++k) {
    p = p * ((z - T3(k - 1)) / (z + T3(k)));

    q = q + (p * lanczos_gamma_approximation<T3>::coefficients[k]);
  }

  return c10::numbers::sqrttau_v<T3> * q;
}

template<typename T1>
T1
lanczos_ln_gamma_approximation(T1 z) {
  using T2 = T1;
  using T3 = numeric_t<T2>;

  auto g = lanczos_gamma_approximation<T3>::s_g;

  if (std::real(z) < T3(-1)) {
    auto p = at::native::special::sin_pi(z);

    constexpr auto is_not_complex = !is_complex_v<T2>;

    if (is_not_complex) {
      p = std::abs(p);
    }

    return c10::numbers::lnpi_v<T3> - std::log(p) - lanczos_ln_gamma_approximation(-T3(1) - z);
  } else {
    auto p = lanczos_binet1p(z);

    constexpr auto is_not_complex = !is_complex_v<T2>;

    if (is_not_complex) {
      p = std::abs(p);
    }

    return std::log(p) + (z + T3(0.5L)) * std::log(z + g + T3(0.5L)) - (z + g + T3(0.5L));
  }
}
}
}
}
}
