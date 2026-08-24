#include <gtest/gtest.h>

#include <c10/util/generic_math.h>
#include <torch/headeronly/util/BFloat16-math.h>
#include <torch/headeronly/util/NumericUtils.h>
#include <cmath>
#include <limits>
namespace torch {
namespace aot_inductor {

TEST(TestMath, TestDivFloor) {
  EXPECT_EQ(c10::div_floor_floating(5., 0.), INFINITY);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., 2.), 2.);
  EXPECT_DOUBLE_EQ(c10::div_floor_floating(5., -2.), -3.);
  EXPECT_EQ(c10::div_floor_integer(5, 2), 2);
  EXPECT_EQ(c10::div_floor_integer(5, -2), -3);
}

TEST(TestMath, TestIsNan) {
  using torch::headeronly::_isnan;
  const float fnan = std::nanf("");

  EXPECT_FALSE(_isnan(1));
  EXPECT_FALSE(_isnan(1.0f));
  EXPECT_TRUE(_isnan(fnan));
  EXPECT_FALSE(_isnan(1.0));
  EXPECT_TRUE(_isnan(std::nan("")));

  EXPECT_FALSE(_isnan(torch::headeronly::complex<float>(1.0f, 2.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::complex<float>(fnan, 2.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::complex<float>(1.0f, fnan)));

  EXPECT_FALSE(_isnan(torch::headeronly::Half(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::Half(fnan)));
  EXPECT_FALSE(_isnan(torch::headeronly::BFloat16(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::BFloat16(fnan)));

  EXPECT_FALSE(_isnan(torch::headeronly::Float8_e5m2(1.0f)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e5m2(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e4m3fn(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e5m2fnuz(fnan)));
  EXPECT_TRUE(_isnan(torch::headeronly::Float8_e4m3fnuz(fnan)));
}

TEST(TestMath, TestIsInf) {
  using torch::headeronly::_isinf;
  const float finf = std::numeric_limits<float>::infinity();

  EXPECT_FALSE(_isinf(1));
  EXPECT_FALSE(_isinf(1.0f));
  EXPECT_TRUE(_isinf(finf));
  EXPECT_TRUE(_isinf(std::numeric_limits<double>::infinity()));

  EXPECT_FALSE(_isinf(torch::headeronly::Half(1.0f)));
  EXPECT_TRUE(_isinf(torch::headeronly::Half(finf)));
  EXPECT_FALSE(_isinf(torch::headeronly::BFloat16(1.0f)));
  EXPECT_TRUE(_isinf(torch::headeronly::BFloat16(finf)));
  EXPECT_TRUE(_isinf(torch::headeronly::Float8_e5m2(finf)));

  // the fn/fnuz formats have no encoding for infinity
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e4m3fn(1.0f)));
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e5m2fnuz(1.0f)));
  EXPECT_FALSE(_isinf(torch::headeronly::Float8_e4m3fnuz(1.0f)));
}

TEST(TestMath, TestIsReducedFloatingPoint) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::Half;
  using torch::headeronly::is_reduced_floating_point;
  using torch::headeronly::is_reduced_floating_point_v;

  static_assert(is_reduced_floating_point_v<Half>);
  static_assert(is_reduced_floating_point_v<BFloat16>);
  static_assert(!is_reduced_floating_point_v<float>);
  EXPECT_TRUE(is_reduced_floating_point<Half>::value);
  EXPECT_TRUE(is_reduced_floating_point<BFloat16>::value);
  EXPECT_FALSE(is_reduced_floating_point<double>::value);

  // Exercise the std:: overloads that dispatch on
  // is_reduced_floating_point_v.
  EXPECT_NEAR(static_cast<float>(std::acos(Half(0.5f))), 1.0472f, 1e-3);
  EXPECT_NEAR(static_cast<float>(std::sqrt(BFloat16(4.0f))), 2.0f, 1e-2);
}

} // namespace aot_inductor
} // namespace torch
