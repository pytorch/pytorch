#include <gtest/gtest.h>

#include <torch/headeronly/util/BFloat16-math.h>

#include <cmath>

TEST(TestBFloat16Math, TestIsReducedFloatingPoint) {
  static_assert(
      torch::headeronly::is_reduced_floating_point_v<torch::headeronly::Half>);
  static_assert(torch::headeronly::is_reduced_floating_point_v<
                torch::headeronly::BFloat16>);
  static_assert(!torch::headeronly::is_reduced_floating_point_v<float>);
  static_assert(torch::headeronly::is_reduced_floating_point<
                torch::headeronly::Half>::value);

  // c10 alias
  static_assert(c10::is_reduced_floating_point_v<c10::Half>);

  // The std math overloads for reduced floating point compile and run.
  torch::headeronly::Half h(4.0f);
  EXPECT_FLOAT_EQ(static_cast<float>(std::sqrt(h)), 2.0f);
}
