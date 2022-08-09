#include <gtest/gtest.h>
#include <ATen/native/special_functions/factorial.h>

template<typename T1>
struct factorial_fixture {
  T1 f0;
  unsigned int n;
  T1 f;
};

const factorial_fixture<double> fixture_0001[51] = {
    {1.0000000000000000, 0, 0.0},
    {1.0000000000000000, 1, 0.0},
    {2.0000000000000000, 2, 0.0},
    {6.0000000000000000, 3, 0.0},
    {24.000000000000000, 4, 0.0},
    {120.00000000000000, 5, 0.0},
    {720.00000000000000, 6, 0.0},
    {5040.0000000000000, 7, 0.0},
    {40320.000000000000, 8, 0.0},
    {362880.00000000000, 9, 0.0},
    {3628800.0000000000, 10, 0.0},
    {39916800.000000000, 11, 0.0},
    {479001600.00000000, 12, 0.0},
    {6227020800.0000000, 13, 0.0},
    {87178291200.000000, 14, 0.0},
    {1307674368000.0000, 15, 0.0},
    {20922789888000.000, 16, 0.0},
    {355687428096000.00, 17, 0.0},
    {6402373705728000.0, 18, 0.0},
    {1.2164510040883200e+17, 19, 0.0},
    {2.4329020081766400e+18, 20, 0.0},
    {5.1090942171709440e+19, 21, 0.0},
    {1.1240007277776077e+21, 22, 0.0},
    {2.5852016738884978e+22, 23, 0.0},
    {6.2044840173323941e+23, 24, 0.0},
    {1.5511210043330986e+25, 25, 0.0},
    {4.0329146112660565e+26, 26, 0.0},
    {1.0888869450418352e+28, 27, 0.0},
    {3.0488834461171387e+29, 28, 0.0},
    {8.8417619937397019e+30, 29, 0.0},
    {2.6525285981219107e+32, 30, 0.0},
    {8.2228386541779224e+33, 31, 0.0},
    {2.6313083693369352e+35, 32, 0.0},
    {8.6833176188118859e+36, 33, 0.0},
    {2.9523279903960416e+38, 34, 0.0},
    {1.0333147966386145e+40, 35, 0.0},
    {3.7199332678990125e+41, 36, 0.0},
    {1.3763753091226346e+43, 37, 0.0},
    {5.2302261746660112e+44, 38, 0.0},
    {2.0397882081197444e+46, 39, 0.0},
    {8.1591528324789768e+47, 40, 0.0},
    {3.3452526613163808e+49, 41, 0.0},
    {1.4050061177528800e+51, 42, 0.0},
    {6.0415263063373834e+52, 43, 0.0},
    {2.6582715747884489e+54, 44, 0.0},
    {1.1962222086548019e+56, 45, 0.0},
    {5.5026221598120892e+57, 46, 0.0},
    {2.5862324151116818e+59, 47, 0.0},
    {1.2413915592536073e+61, 48, 0.0},
    {6.0828186403426752e+62, 49, 0.0},
    {3.0414093201713376e+64, 50, 0.0},
};

const double tolerance_0001 = 2.5000000000000020e-13;

template<typename T1, unsigned int T2>
void
test(const factorial_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::factorial<T1>(fixture.n);

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