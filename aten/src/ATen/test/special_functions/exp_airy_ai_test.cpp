#include <gtest/gtest.h>
#include <ATen/native/special_functions/exp_airy_ai.h>

template<typename T1>
struct exp_airy_ai_fixture {
  T1 f0;
  T1 x;
  T1 f;
};

const exp_airy_ai_fixture<double> fixture_0001[21] = {
    {0.35502805388781722, 0.0000000000000000, 0.0},
    {0.29327715912994734, 0.50000000000000000, 0.0},
    {0.26351364474914007, 1.0000000000000000, 0.0},
    {0.24418489767140847, 1.5000000000000000, 0.0},
    {0.23016491865251160, 2.0000000000000000, 0.0},
    {0.21932220512871203, 2.5000000000000000, 0.0},
    {0.21057204278597699, 3.0000000000000000, 0.0},
    {0.20329208081635175, 3.5000000000000000, 0.0},
    {0.19709480264306647, 4.0000000000000000, 0.0},
    {0.19172396872398537, 4.5000000000000000, 0.0},
    {0.18700211893594343, 5.0000000000000000, 0.0},
    {0.18280173946240377, 5.5000000000000000, 0.0},
    {0.17902840741321008, 6.0000000000000000, 0.0},
    {0.17561043019266195, 6.5000000000000000, 0.0},
    {0.17249220797740278, 7.0000000000000000, 0.0},
    {0.16962983096364936, 7.5000000000000000, 0.0},
    {0.16698807106393279, 8.0000000000000000, 0.0},
    {0.16453827306790608, 8.5000000000000000, 0.0},
    {0.16225684290423317, 9.0000000000000000, 0.0},
    {0.16012414238108222, 9.5000000000000000, 0.0},
    {0.15812366685434615, 10.000000000000000, 0.0},
};

const double tolerance_0001 = 5.0000000000000039e-13;

template<typename T1, unsigned int T2>
void
test(const exp_airy_ai_fixture<T1>(&fixtures)[T2], T1 tolerance) {
  const T1 epsilon = std::numeric_limits<T1>::epsilon();

  T1 maximum_absolute_difference = T1(-1);
  T1 maximum_absolute_fraction = T1(-1);

  auto error = 0;

  for (auto fixture: fixtures) {
    const T1 g = aten::native::special_functions::exp_airy_ai(fixture.x);

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

TEST(ExpAiryAiTest, GSL) {
  test(fixture_0001, tolerance_0001);
}