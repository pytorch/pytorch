#include <gtest/gtest.h>
#include <ATen/native/special_functions/exp_modified_bessel_i.h>

template<typename T1>
struct exp_modified_bessel_i_fixture {
  T1 f0;
  T1 n;
  T1 x;
  T1 f;
};

const exp_modified_bessel_i_fixture<double> fixture_0001[11] = {
    {8.5155875815486127e-05, 100.00000000000000, 1000.0000000000000, 0.0},
    {0.00012783770668059490, 100.00000000000000, 1100.0000000000000, 0.0},
    {0.00017868879940604636, 100.00000000000000, 1200.0000000000000, 0.0},
    {0.00023648188537317519, 100.00000000000000, 1300.0000000000000, 0.0},
    {0.00029987385710643594, 100.00000000000000, 1400.0000000000000, 0.0},
    {0.00036754116875294015, 100.00000000000000, 1500.0000000000000, 0.0},
    {0.00043825961543828511, 100.00000000000000, 1600.0000000000000, 0.0},
    {0.00051094422898306715, 100.00000000000000, 1700.0000000000000, 0.0},
    {0.00058466302033197875, 100.00000000000000, 1800.0000000000000, 0.0},
    {0.00065863487030955832, 100.00000000000000, 1900.0000000000000, 0.0},
    {0.00073221866792298469, 100.00000000000000, 2000.0000000000000, 0.0},
};

const double tolerance_0001 = 1.0000000000000006e-10;

template<typename T1, unsigned int T2>
void
test(const exp_modified_bessel_i_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::exp_modified_bessel_i(fixture.n, fixture.x);

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

TEST(ExpModifiedBesselITest, GSL) {
  test(fixture_0001, tolerance_0001);
}