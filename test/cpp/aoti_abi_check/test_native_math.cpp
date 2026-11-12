#include <gtest/gtest.h>

#include <torch/headeronly/native/Math.h>

#include <cmath>
#include <cstddef>

TEST(TestNativeMath, TestCalcErfinv) {
  EXPECT_NEAR(calc_erfinv(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(calc_erfinv(0.5f), 0.47693627620446987f, 1e-5f);
  EXPECT_NEAR(calc_erfinv(0.9), 1.1630871536766743, 1e-13);
  EXPECT_NEAR(calc_erfinv(-0.9), -1.1630871536766743, 1e-13);
  EXPECT_NEAR(
      static_cast<float>(calc_erfinv(torch::headeronly::BFloat16{0.5f})),
      0.47693627620446987f,
      5e-3f);
}

TEST(TestNativeMath, TestCalcI0e) {
  EXPECT_NEAR(calc_i0e(0.0f), 1.0f, 1e-5f);
  EXPECT_NEAR(calc_i0e(1.0f), 0.46575960759364043f, 1e-5f);
  EXPECT_NEAR(calc_i0e(1.0), 0.46575960759364043, 1e-12);
}

TEST(TestNativeMath, TestCalcDigamma) {
  EXPECT_NEAR(calc_digamma(1.0f), -0.5772156649015329f, 1e-5f);
  EXPECT_NEAR(calc_digamma(-0.5), 0.03648997397857652, 1e-12);
  EXPECT_NEAR(
      static_cast<float>(calc_digamma(torch::headeronly::BFloat16{1.0f})),
      -0.5772156649015329f,
      5e-3f);
  EXPECT_NEAR(
      static_cast<float>(calc_digamma(torch::headeronly::Half{1.0f})),
      -0.5772156649015329f,
      1e-3f);
}

TEST(TestNativeMath, TestChbevl) {
  static const float coeff[] = {1.0f, 2.0f, 3.0f};
  EXPECT_NEAR(chbevl(0.0f, coeff, 3), 0.5f, 1e-6f);
  EXPECT_NEAR(chbevl(0.0f, coeff, std::size_t{3}), 0.5f, 1e-6f);
  static const double double_coeff[] = {1.0, 2.0, 3.0};
  EXPECT_NEAR(chbevl(0.0, double_coeff, 3), 0.5, 1e-12);
  EXPECT_NEAR(chbevl(0.0, double_coeff, std::size_t{3}), 0.5, 1e-12);
}

TEST(TestNativeMath, TestExp2Impl) {
  EXPECT_FLOAT_EQ(exp2_impl(2.0f), 4.0f);
  EXPECT_DOUBLE_EQ(exp2_impl(2.0), 4.0);

  const auto complex_float =
      exp2_impl(torch::headeronly::complex<float>{1.0f, 1.0f});
  EXPECT_NEAR(complex_float.real(), 2.0f * std::cos(std::log(2.0f)), 1e-6f);
  EXPECT_NEAR(complex_float.imag(), 2.0f * std::sin(std::log(2.0f)), 1e-6f);

  const auto complex_double =
      exp2_impl(torch::headeronly::complex<double>{1.0, 1.0});
  EXPECT_NEAR(complex_double.real(), 2.0 * std::cos(std::log(2.0)), 1e-12);
  EXPECT_NEAR(complex_double.imag(), 2.0 * std::sin(std::log(2.0)), 1e-12);
}

TEST(TestNativeMath, TestCalcI0) {
  EXPECT_NEAR(calc_i0(0.0f), 1.0f, 1e-5f);
  EXPECT_NEAR(calc_i0(1.0), 1.2660658777520084, 1e-12);
  EXPECT_NEAR(
      static_cast<float>(calc_i0(torch::headeronly::BFloat16{1.0f})),
      1.2660658777520084f,
      5e-3f);
  EXPECT_NEAR(
      static_cast<float>(calc_i0(torch::headeronly::Half{1.0f})),
      1.2660658777520084f,
      2e-3f);
}

TEST(TestNativeMath, TestCalcIgamma) {
  EXPECT_NEAR(calc_igamma(1.0f, 1.0f), 0.6321205588285577f, 1e-5f);
  EXPECT_NEAR(calc_igammac(1.0f, 1.0f), 0.36787944117144233f, 1e-5f);
  EXPECT_NEAR(calc_igamma(1.0, 1.0), 0.6321205588285577, 1e-12);
  EXPECT_NEAR(calc_igammac(1.0, 1.0), 0.36787944117144233, 1e-12);

  const auto bfloat_one = torch::headeronly::BFloat16{1.0f};
  EXPECT_NEAR(
      static_cast<float>(calc_igamma(bfloat_one, bfloat_one)),
      0.6321205588285577f,
      5e-3f);
  EXPECT_NEAR(
      static_cast<float>(calc_igammac(bfloat_one, bfloat_one)),
      0.36787944117144233f,
      5e-3f);

  const auto half_one = torch::headeronly::Half{1.0f};
  EXPECT_NEAR(
      static_cast<float>(calc_igamma(half_one, half_one)),
      0.6321205588285577f,
      2e-3f);
  EXPECT_NEAR(
      static_cast<float>(calc_igammac(half_one, half_one)),
      0.36787944117144233f,
      2e-3f);
}

TEST(TestNativeMath, TestPolevl) {
  static const float coeff[] = {1.0f, 1.0f, 1.0f};
  EXPECT_NEAR(polevl(1.0f, coeff, 2), 3.0f, 1e-6f);
  static const double double_coeff[] = {1.0, 1.0, 1.0};
  EXPECT_NEAR(polevl(1.0, double_coeff, 2), 3.0, 1e-12);
}
