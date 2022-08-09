#pragma once

#include <cmath>

#include <c10/util/numbers.h>

namespace at::native::special_functions::detail {
template<typename T1>
T1
znorm2(T1 x) {
  return T1{0.5L} * std::erfc(x / c10::numbers::sqrt2_v<T1>);
}

template<typename T1>
T1
znorm1(T1 x) {
  return T1{0.5L} * std::erf(x / c10::numbers::sqrt2_v<T1>);
}

template<typename T1>
T1
owens_t(T1 h, T1 a) {
  constexpr std::size_t s_num_c2 = 21;
  constexpr T1 s_c2[s_num_c2] = {
      0.99999999999999987510L,
      -0.99999999999988796462L,
      0.99999999998290743652L,
      -0.99999999896282500134L,
      0.99999996660459362918L,
      -0.99999933986272476760L,
      0.99999125611136965852L,
      -0.99991777624463387686L,
      0.99942835555870132569L,
      -0.99697311720723000295L,
      0.98751448037275303682L,
      -0.95915857980572882813L,
      0.89246305511006708555L,
      -0.76893425990463999675L,
      0.58893528468484693250L,
      -0.38380345160440256652L,
      0.20317601701045299653L,
      -0.82813631607004984866e-01L,
      0.24167984735759576523e-01L,
      -0.44676566663971825242e-02L,
      0.39141169402373836468e-03L
  };

  constexpr T1 s_h_range[14]
      {
          0.02L, 0.06L, 0.09L, 0.125L, 0.26L, 0.4L, 0.6L,
          1.6L, 1.7L, 2.33L, 2.4L, 3.36L, 3.4L, 4.8L
      };

  constexpr T1 s_a_range[7]
      {
          0.025L,
          0.09L,
          0.15L,
          0.36L,
          0.5L,
          0.9L,
          0.99999L
      };

  constexpr int s_select[8][15]
      {
          {0, 0, 1, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 8},
          {0, 1, 1, 2, 2, 4, 4, 13, 13, 14, 14, 15, 15, 15, 8},
          {1, 1, 2, 2, 2, 4, 4, 14, 14, 14, 14, 15, 15, 15, 9},
          {1, 1, 2, 4, 4, 4, 4, 6, 6, 15, 15, 15, 15, 15, 9},
          {1, 2, 2, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 10},
          {1, 2, 4, 4, 4, 5, 5, 7, 7, 16, 16, 16, 11, 11, 11},
          {1, 2, 3, 3, 5, 5, 7, 7, 16, 16, 16, 16, 16, 11, 11},
          {1, 2, 3, 3, 5, 5, 17, 17, 17, 17, 16, 16, 16, 11, 11}
      };

  constexpr T1 GJ[13]
      {
          0.35082039676451715489e-02L,
          0.31279042338030753740e-01L,
          0.85266826283219451090e-01L,
          0.16245071730812277011L,
          0.25851196049125434828L,
          0.36807553840697533536L,
          0.48501092905604697475L,
          0.60277514152618576821L,
          0.71477884217753226516L,
          0.81475510988760098605L,
          0.89711029755948965867L,
          0.95723808085944261843L,
          0.99178832974629703586L
      };

  constexpr T1 s_GJ_wts[13]
      {
          0.18831438115323502887e-01L,
          0.18567086243977649478e-01L,
          0.18042093461223385584e-01L,
          0.17263829606398753364e-01L,
          0.16243219975989856730e-01L,
          0.14994592034116704829e-01L,
          0.13535474469662088392e-01L,
          0.11886351605820165233e-01L,
          0.10070377242777431897e-01L,
          0.81130545742299586629e-02L,
          0.60419009528470238773e-02L,
          0.38862217010742057883e-02L,
          0.16793031084546090448e-02L
      };

  if (std::isnan(a) || std::isnan(h))
    return std::numeric_limits<T1>::quiet_NaN();
  else if (h < T1(0))
    return owens_t(-h, a);
  else if (a < T1(0))
    return -owens_t(h, -a);
  else if (a > T1(1)) {
    return T1{0.5L} * znorm2(h) + T1{0.5L} * znorm2(a * h) - znorm2(h) * znorm2(a * h) - owens_t(h * a, T1(1) / a);
  }
  if (h == T1(0))
    return std::atan(a) * T1{0.15915494309189533577L};
  if (a == T1(0))
    return T1(0);
  if (a == T1(1))
    return T1{0.5L} * znorm2(-h) * znorm2(h);
  else {
    //  Determine appropriate method from t1...t6

    auto iaint = 7;
    for (int i = 0; i < 7; ++i)
      if (a <= s_a_range[i]) {
        iaint = i;
        break;
      }

    auto ihint = 14;
    for (int i = 0; i < 14; ++i)
      if (h <= s_h_range[i]) {
        ihint = i;
        break;
      }

    auto icode = s_select[iaint][ihint];

    constexpr int ORDERS[18] = {2, 3, 4, 5, 7, 10, 12, 18, 10, 20, 30, 20, 4, 7, 8, 20, 13, 0};

    auto m = ORDERS[icode];

    constexpr int METHODS[18] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6};

    if (METHODS[icode] == 1) {
      auto j = 1;
      auto jj = T1(1);
      auto aj = T1{0.15915494309189533577L} * a;
      auto dj = std::expm1(-T1{0.5L} * h * h);
      auto gj = -T1{0.5L} * h * h * std::exp(-T1{0.5L} * h * h);

      auto value = T1{0.15915494309189533577L} * std::atan(a);
      while (true) {
        auto z = dj * aj / jj;
        value += z;

        if (j >= m && std::abs(z) < std::numeric_limits<T1>::epsilon() * std::abs(value))
          return value;

        ++j;
        jj += T1(2);
        aj *= a * a;
        dj = gj - dj;
        gj *= -T1{0.5L} * h * h / T1(j);
      }
    } else if (METHODS[icode] == 2) {
      auto maxii = m + m + 1;
      auto ii = 1;
      auto ah = a * h;
      auto vi = T1{0.39894228040143267794L} * a * std::exp(-T1{0.5} * ah * ah);
      auto z = znorm1(ah) / h;
      auto z_prev = std::abs(z) + T1(1);

      auto value = T1(0);
      while (true) {
        value += z;

        if (std::abs(z) < std::numeric_limits<T1>::epsilon() * std::abs(value)
            || (ii >= maxii && std::abs(z) > z_prev)
            || std::abs(z) < std::numeric_limits<T1>::epsilon()) {
          value *= T1{0.39894228040143267794L} * std::exp(-T1{0.5} * (h * h));
          return value;
        }

        z_prev = std::abs(z);
        z = T1(1) / (h * h) * (vi - T1(ii) * z);
        vi *= -a * a;
        ii += 2;
      }
    } else if (METHODS[icode] == 3) {
      auto ii = 1;
      auto vi = T1{0.39894228040143267794L} * a * std::exp(-T1{0.5L} * (a * h) * (a * h));
      auto zi = znorm1(a * h) / h;

      auto value = T1(0);
      for (auto i = 0ull; i < s_num_c2; ++i) {
        value += zi * s_c2[i];
        zi = T1(1) / (h * h) * (T1(ii) * zi - vi);
        vi *= a * a;
        ii += 2;
      }

      return value * (T1{0.39894228040143267794L} * std::exp(-T1{0.5L} * (h * h)));
    } else if (METHODS[icode] == 4) {
      auto ii = 1;
      auto ai = T1{0.15915494309189533577L} * a * std::exp(-T1{0.5L} * (h * h) * (T1(1) - -a * a));
      auto yi = T1(1);

      auto value = ai * yi;
      while (true) {
        ii += 2;
        yi = (T1(1) - h * h * yi) / T1(ii);
        ai *= -a * a;
        auto z = ai * yi;
        value += z;

        if (std::abs(z) > std::numeric_limits<T1>::min()
            && std::abs(z) < std::numeric_limits<T1>::epsilon() * std::abs(value)) {
          return value;
        }
      }
    } else if (METHODS[icode] == 5) {
      auto p = T1(0);

      for (unsigned long long j = 0; j < 13; j++) {
        p = p + (s_GJ_wts[j] * std::exp(-T1{0.5L} * h * h * (T1(1) + a * a * GJ[j])) / (T1(1) + a * a * GJ[j]));
      }

      return p * a;
    } else if (METHODS[icode] == 6) {
      if (std::abs(std::atan2(T1(1) - a, T1(1) + a)) > std::numeric_limits<T1>::epsilon()) {
        return T1{0.5L} * znorm2(h) * (T1(1) - znorm2(h))
            - (T1{0.15915494309189533577L}
                * std::atan2(T1(1) - a, T1(1) + a)
                * std::exp(-T1{0.5L} * (T1(1) - a) * h * h / std::atan2(T1(1) - a, T1(1) + a)));
      } else {
        return T1{0.5L} * znorm2(h) * (T1(1) - znorm2(h));
      }
    }

    return T1(0);
  }
}
}
