#include <gtest/gtest.h>

#include <torch/headeronly/util/FloatRounding.h>

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

} // namespace aot_inductor
} // namespace torch
