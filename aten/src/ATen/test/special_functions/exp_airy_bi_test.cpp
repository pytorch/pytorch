#include <gtest/gtest.h>
#include <ATen/native/special_functions/exp_airy_bi.h>

template<typename T1>
struct exp_airy_bi_fixture {
  T1 f0;
  T1 x;
  T1 f;
};

const exp_airy_bi_fixture<double> fixture_0001[21] = {
    {0.61492662744600068, 0.0000000000000000, 0.0},
    {0.67489241111563025, 0.50000000000000000, 0.0},
    {0.61991194357267854, 1.0000000000000000, 0.0},
    {0.55209437228578417, 1.5000000000000000, 0.0},
    {0.50043725430409502, 2.0000000000000000, 0.0},
    {0.46475048019609250, 2.5000000000000000, 0.0},
    {0.43938402355009643, 3.0000000000000000, 0.0},
    {0.42017718823530570, 3.5000000000000000, 0.0},
    {0.40480946788929806, 4.0000000000000000, 0.0},
    {0.39202734094459063, 4.5000000000000000, 0.0},
    {0.38110853108887738, 5.0000000000000000, 0.0},
    {0.37160000660099485, 5.5000000000000000, 0.0},
    {0.36319693054542684, 6.0000000000000000, 0.0},
    {0.35568337591227883, 6.5000000000000000, 0.0},
    {0.34890049029582110, 7.0000000000000000, 0.0},
    {0.34272793654369132, 7.5000000000000000, 0.0},
    {0.33707237582041633, 8.0000000000000000, 0.0},
    {0.33185997650946220, 8.5000000000000000, 0.0},
    {0.32703135827743030, 9.0000000000000000, 0.0},
    {0.32253807502213006, 9.5000000000000000, 0.0},
    {0.31834010533673446, 10.000000000000000, 0.0},
};

const double tolerance_0001 = 0.050000000000000003;

template<typename T1, unsigned int T2>
void
test(const exp_airy_bi_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::exp_airy_bi(fixture.x);

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

TEST(ExpAiryBiTest, GSL) {
  test(fixture_0001, tolerance_0001);
}