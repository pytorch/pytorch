#include <gtest/gtest.h>

#include <torch/headeronly/util/FloatRounding.h>

#include <cmath>
#include <limits>

namespace torch {
namespace aot_inductor {

TEST(TestFloatRounding, TestConvertWithRounding) {
  using torch::headeronly::convert_with_rounding;
  using torch::headeronly::Float8_e8m0fnu;
  using torch::headeronly::Half;
  using torch::headeronly::RoundingMode;

  // 1 + 2^-11 + 2^-40 is just above the midpoint between 1 and the next
  // float16, 1 + 2^-10.
  const double x = 1.0 + 0x1p-11 + 0x1p-40;
  EXPECT_EQ(
      static_cast<float>(
          convert_with_rounding<Half>(x, RoundingMode::ToNearestEven)),
      1.0f + 0x1p-10f);
  EXPECT_EQ(
      static_cast<float>(
          convert_with_rounding<Half>(x, RoundingMode::TowardZero)),
      1.0f);
  EXPECT_EQ(
      static_cast<float>(
          convert_with_rounding<Half>(-x, RoundingMode::TowardPositive)),
      -1.0f);
  EXPECT_EQ(
      static_cast<float>(
          convert_with_rounding<Half>(-x, RoundingMode::TowardNegative)),
      -1.0f - 0x1p-10f);
  EXPECT_EQ(
      static_cast<float>(convert_with_rounding<Float8_e8m0fnu>(
          2.5f, RoundingMode::ToNearestEven)),
      2.0f);
  EXPECT_EQ(
      static_cast<float>(convert_with_rounding<Float8_e8m0fnu>(
          2.5f, RoundingMode::TowardPositive)),
      4.0f);
}

// Every other destination: one rounding, then overflow, infinity and NaN, which
// depend on the format.
TEST(TestFloatRounding, TestConvertWithRoundingFormats) {
  using namespace torch::headeronly;
  constexpr auto kNearest = RoundingMode::ToNearestEven;
  constexpr auto kZero = RoundingMode::TowardZero;
  constexpr float kInfinity = std::numeric_limits<float>::infinity();
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const auto value = [](auto v) { return static_cast<float>(v); };

  // Just above midpoints.
  const double d = 1.0 + 0x1p-24 + 0x1p-40;
  EXPECT_EQ(convert_with_rounding<float>(d, kNearest), 1.0f + 0x1p-23f);
  EXPECT_EQ(convert_with_rounding<float>(d, kZero), 1.0f);
  const float f = 1.0f + 0x1p-8f + 0x1p-20f;
  EXPECT_EQ(
      value(convert_with_rounding<BFloat16>(f, kNearest)), 1.0f + 0x1p-7f);
  EXPECT_EQ(value(convert_with_rounding<BFloat16>(f, kZero)), 1.0f);

  // Saturates.
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e4m3fn>(1e6f, kNearest)), 448.0f);
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e4m3fn>(-kInfinity, kZero)), -448.0f);
  EXPECT_TRUE(
      std::isnan(value(convert_with_rounding<Float8_e4m3fn>(kNaN, kZero))));
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e5m2>(1e6f, kNearest)), 57344.0f);
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e5m2>(-kInfinity, kZero)), -57344.0f);
  // NaN on overflow, and no negative zero.
  EXPECT_TRUE(std::isnan(
      value(convert_with_rounding<Float8_e4m3fnuz>(1e6f, kNearest))));
  EXPECT_EQ(value(convert_with_rounding<Float8_e4m3fnuz>(1e6f, kZero)), 240.0f);
  EXPECT_EQ(convert_with_rounding<Float8_e4m3fnuz>(-0.0f, kNearest).x, 0);
  EXPECT_EQ(convert_with_rounding<Float8_e4m3fnuz>(-0x1p-12f, kNearest).x, 0);
  EXPECT_TRUE(std::isnan(
      value(convert_with_rounding<Float8_e5m2fnuz>(kInfinity, kZero))));
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e5m2fnuz>(1e6f, kZero)), 57344.0f);
  EXPECT_EQ(convert_with_rounding<Float8_e5m2fnuz>(-0.0f, kNearest).x, 0);
  EXPECT_EQ(convert_with_rounding<Float8_e5m2fnuz>(-0x1p-19f, kNearest).x, 0);
  // Unsigned: the magnitude is rounded, and saturates.
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e5m3fnu>(
          -2.1f, RoundingMode::TowardNegative)),
      2.0f);
  EXPECT_EQ(
      value(convert_with_rounding<Float8_e5m3fnu>(kInfinity, kNearest)),
      114688.0f);
  EXPECT_TRUE(
      std::isnan(value(convert_with_rounding<Float8_e5m3fnu>(kNaN, kZero))));
}

} // namespace aot_inductor
} // namespace torch
