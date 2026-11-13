#include <gtest/gtest.h>

#include <torch/headeronly/util/MathConstants.h>

#include <cmath>

namespace torch {
namespace aot_inductor {

TEST(TestMathConstants, TestDoubleConstants) {
  using torch::headeronly::e;
  using torch::headeronly::euler;
  using torch::headeronly::frac_1_pi;
  using torch::headeronly::frac_1_sqrt_pi;
  using torch::headeronly::frac_sqrt_2;
  using torch::headeronly::frac_sqrt_3;
  using torch::headeronly::golden_ratio;
  using torch::headeronly::ln_10;
  using torch::headeronly::ln_2;
  using torch::headeronly::log_10_e;
  using torch::headeronly::log_2_e;
  using torch::headeronly::pi;
  using torch::headeronly::sqrt_2;
  using torch::headeronly::sqrt_3;

  EXPECT_NEAR(pi<double>, 3.14159265358979, 1e-12);
  EXPECT_NEAR(e<double>, 2.71828182845905, 1e-12);
  EXPECT_NEAR(euler<double>, 0.57721566490153, 1e-12);
  EXPECT_NEAR(sqrt_2<double>, 1.41421356237310, 1e-12);
  EXPECT_NEAR(sqrt_3<double>, 1.73205080756888, 1e-12);
  EXPECT_NEAR(frac_1_pi<double>, 1.0 / pi<double>, 1e-12);
  EXPECT_NEAR(frac_1_sqrt_pi<double>, 1.0 / std::sqrt(pi<double>), 1e-12);
  EXPECT_NEAR(frac_sqrt_2<double>, 1.0 / sqrt_2<double>, 1e-12);
  EXPECT_NEAR(frac_sqrt_3<double>, 1.0 / sqrt_3<double>, 1e-12);
  EXPECT_NEAR(golden_ratio<double>, 1.61803398874989, 1e-12);
  EXPECT_NEAR(ln_10<double>, std::log(10.0), 1e-12);
  EXPECT_NEAR(ln_2<double>, std::log(2.0), 1e-12);
  EXPECT_NEAR(log_10_e<double>, 1.0 / ln_10<double>, 1e-12);
  EXPECT_NEAR(log_2_e<double>, 1.0 / ln_2<double>, 1e-12);
}

TEST(TestMathConstants, TestReducedPrecisionSpecializations) {
  using torch::headeronly::BFloat16;
  using torch::headeronly::Half;
  using torch::headeronly::pi;

  // Bit-exact specializations;
  EXPECT_EQ((pi<BFloat16>).x, 0x4049);
  EXPECT_EQ((pi<Half>).x, 0x4248);
}

} // namespace aot_inductor
} // namespace torch
