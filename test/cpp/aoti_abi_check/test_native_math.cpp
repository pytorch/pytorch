#include <gtest/gtest.h>

#include <torch/headeronly/native/Math.h>

TEST(TestNativeMath, TestCalcErfinv) {
  EXPECT_NEAR(calc_erfinv(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(calc_erfinv(0.5f), 0.47693627620446987f, 1e-5f);
}

TEST(TestNativeMath, TestCalcI0e) {
  EXPECT_NEAR(calc_i0e(0.0f), 1.0f, 1e-5f);
  EXPECT_NEAR(calc_i0e(1.0f), 0.6975650823136145f, 1e-5f);
}

TEST(TestNativeMath, TestCalcDigamma) {
  EXPECT_NEAR(calc_digamma(1.0f), -0.5772156649015329f, 1e-5f);
}

TEST(TestNativeMath, TestChbevl) {
  static const float coeff[] = {1.0f, 2.0f, 3.0f};
  EXPECT_NEAR(chbevl(0.0f, coeff, 3), 2.0f, 1e-6f);
}

TEST(TestNativeMath, TestExp2Impl) {
  EXPECT_FLOAT_EQ(exp2_impl(2.0f), 4.0f);
}

TEST(TestNativeMath, TestCalcGcd) {
  EXPECT_EQ(calc_gcd(12, 8), 4);
}

TEST(TestNativeMath, TestCalcI0) {
  EXPECT_NEAR(calc_i0(0.0f), 1.0f, 1e-5f);
}

TEST(TestNativeMath, TestCalcI1) {
  EXPECT_NEAR(calc_i1(0.0f), 0.0f, 1e-5f);
  EXPECT_NEAR(calc_i1e(1.0f), 0.781212112003936f, 1e-5f);
}

TEST(TestNativeMath, TestCalcIgamma) {
  EXPECT_NEAR(calc_igamma(1.0f, 1.0f), 0.6321205588285577f, 1e-5f);
  EXPECT_NEAR(calc_igammac(1.0f, 1.0f), 0.36787944117144233f, 1e-5f);
}

TEST(TestNativeMath, TestPolevlAndZeta) {
  static const float coeff[] = {1.0f, 1.0f, 1.0f};
  EXPECT_NEAR(polevl(1.0f, coeff, 2), 3.0f, 1e-6f);
  EXPECT_GT(zeta(2.0f, 1.0f), 1.0f);
}
