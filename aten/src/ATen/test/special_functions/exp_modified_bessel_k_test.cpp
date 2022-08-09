#include <gtest/gtest.h>
#include <ATen/native/special_functions/exp_modified_bessel_k.h>

template<typename T1>
struct exp_modified_bessel_k_fixture {
  T1 f0;
  T1 n;
  T1 x;
  T1 f;
};

const exp_modified_bessel_k_fixture<double> fixture_0001[11] = {
    {5.8424465521265159, 100.00000000000000, 1000.0000000000000, 0.0},
    {3.5410426710741936, 100.00000000000000, 1100.0000000000000, 0.0},
    {2.3237462838842249, 100.00000000000000, 1200.0000000000000, 0.0},
    {1.6216147846822155, 100.00000000000000, 1300.0000000000000, 0.0},
    {1.1879504112001300, 100.00000000000000, 1400.0000000000000, 0.0},
    {0.90491922756991028, 100.00000000000000, 1500.0000000000000, 0.0},
    {0.71165910479993821, 100.00000000000000, 1600.0000000000000, 0.0},
    {0.57464221239235203, 100.00000000000000, 1700.0000000000000, 0.0},
    {0.47437600624377624, 100.00000000000000, 1800.0000000000000, 0.0},
    {0.39899827109763819, 100.00000000000000, 1900.0000000000000, 0.0},
    {0.34100208493029188, 100.00000000000000, 2000.0000000000000, 0.0},
};

const double tolerance_0001 = 2.5000000000000020e-13;

template<typename T1, unsigned int T2>
void
test(const exp_modified_bessel_k_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::exp_modified_bessel_k(fixture.n, fixture.x);

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

TEST(ExpModifiedBesselKTest, GSL) {
  test(fixture_0001, tolerance_0001);
}