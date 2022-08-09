#include <gtest/gtest.h>
#include <ATen/native/special_functions/complete_elliptic_integral_e.h>

template<typename T1>
struct complete_elliptic_integral_e_fixture {
  T1 f0;
  T1 k;
  T1 f;
};

const complete_elliptic_integral_e_fixture<double> fixture_0001[19] = {
    {1.1716970527816142, -0.90000000000000002, 0.0},
    {1.2763499431699064, -0.80000000000000004, 0.0},
    {1.3556611355719554, -0.69999999999999996, 0.0},
    {1.4180833944487243, -0.59999999999999998, 0.0},
    {1.4674622093394272, -0.50000000000000000, 0.0},
    {1.5059416123600404, -0.39999999999999991, 0.0},
    {1.5348334649232491, -0.29999999999999993, 0.0},
    {1.5549685462425293, -0.19999999999999996, 0.0},
    {1.5668619420216683, -0.099999999999999978, 0.0},
    {1.5707963267948966, 0.0000000000000000, 0.0},
    {1.5668619420216683, 0.10000000000000009, 0.0},
    {1.5549685462425293, 0.20000000000000018, 0.0},
    {1.5348334649232491, 0.30000000000000004, 0.0},
    {1.5059416123600402, 0.40000000000000013, 0.0},
    {1.4674622093394272, 0.50000000000000000, 0.0},
    {1.4180833944487241, 0.60000000000000009, 0.0},
    {1.3556611355719554, 0.70000000000000018, 0.0},
    {1.2763499431699064, 0.80000000000000004, 0.0},
    {1.1716970527816140, 0.90000000000000013, 0.0},
};

const double tolerance_0001 = 2.5000000000000020e-13;

template<typename T1, unsigned int T2>
void
test(const complete_elliptic_integral_e_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::complete_elliptic_integral_e(fixture.k);

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

TEST(CompleteEllipticIntegralETest, GSL) {
  test(fixture_0001, tolerance_0001);
}