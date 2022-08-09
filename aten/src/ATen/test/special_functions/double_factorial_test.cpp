#include <gtest/gtest.h>
#include <ATen/native/special_functions/double_factorial.h>

template<typename T1>
struct double_factorial_fixture {
  T1 f0;
  unsigned int n;
  T1 f;
};

const double_factorial_fixture<double> fixture_0001[51] = {
    {1.0000000000000000, 0, 0.0},
    {1.0000000000000000, 1, 0.0},
    {2.0000000000000000, 2, 0.0},
    {3.0000000000000000, 3, 0.0},
    {8.0000000000000000, 4, 0.0},
    {15.000000000000000, 5, 0.0},
    {48.000000000000000, 6, 0.0},
    {105.00000000000000, 7, 0.0},
    {384.00000000000000, 8, 0.0},
    {945.00000000000000, 9, 0.0},
    {3840.0000000000000, 10, 0.0},
    {10395.000000000000, 11, 0.0},
    {46080.000000000000, 12, 0.0},
    {135135.00000000000, 13, 0.0},
    {645120.00000000000, 14, 0.0},
    {2027025.0000000000, 15, 0.0},
    {10321920.000000000, 16, 0.0},
    {34459425.000000000, 17, 0.0},
    {185794560.00000000, 18, 0.0},
    {654729075.00000000, 19, 0.0},
    {3715891200.0000000, 20, 0.0},
    {13749310575.000000, 21, 0.0},
    {81749606400.000000, 22, 0.0},
    {316234143225.00000, 23, 0.0},
    {1961990553600.0000, 24, 0.0},
    {7905853580625.0000, 25, 0.0},
    {51011754393600.000, 26, 0.0},
    {213458046676875.00, 27, 0.0},
    {1428329123020800.0, 28, 0.0},
    {6190283353629375.0, 29, 0.0},
    {42849873690624000., 30, 0.0},
    {1.9189878396251062e+17, 31, 0.0},
    {1.3711959580999680e+18, 32, 0.0},
    {6.3326598707628503e+18, 33, 0.0},
    {4.6620662575398912e+19, 34, 0.0},
    {2.2164309547669976e+20, 35, 0.0},
    {1.6783438527143608e+21, 36, 0.0},
    {8.2007945326378919e+21, 37, 0.0},
    {6.3777066403145712e+22, 38, 0.0},
    {3.1983098677287775e+23, 39, 0.0},
    {2.5510826561258285e+24, 40, 0.0},
    {1.3113070457687988e+25, 41, 0.0},
    {1.0714547155728480e+26, 42, 0.0},
    {5.6386202968058351e+26, 43, 0.0},
    {4.7144007485205310e+27, 44, 0.0},
    {2.5373791335626256e+28, 45, 0.0},
    {2.1686243443194444e+29, 46, 0.0},
    {1.1925681927744342e+30, 47, 0.0},
    {1.0409396852733332e+31, 48, 0.0},
    {5.8435841445947271e+31, 49, 0.0},
    {5.2046984263666663e+32, 50, 0.0},
};

const double tolerance_0001 = 0.00050000000000000001;

template<typename T1, unsigned int T2>
void
test(const double_factorial_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::double_factorial<T1>(fixture.n);

    if (std::isnan(g) && !error) {
      error = 1;
    }

    if (!std::isnan(g)) {
      const T1 f = fixture.f0;

      const auto difference = g - f;

      const auto absolute_difference = std::abs(difference);

      if (absolute_difference > maximum_absolute_difference) {
        maximum_absolute_difference = absolute_difference;
      }

      const auto abs_f = std::abs(f);
      const auto abs_g = std::abs(g);

      if (abs_f > T1(10) * epsilon && abs_g > T1(10) * epsilon) {
        const auto fraction = difference / f;

        const auto absolute_fraction = std::abs(fraction);

        if (absolute_fraction > maximum_absolute_fraction) {
          maximum_absolute_fraction = absolute_fraction;
        }
      }
    }
  }

  EXPECT_TRUE(!error && maximum_absolute_fraction < tolerance);
}
