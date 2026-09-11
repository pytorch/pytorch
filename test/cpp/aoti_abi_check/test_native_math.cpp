#include <gtest/gtest.h>

#include <torch/headeronly/native/Math.h>

TEST(TestNativeMath, TestCalcErfinv) {
  EXPECT_NEAR(calc_erfinv(0.0f), 0.0f, 1e-6f);
  EXPECT_NEAR(calc_erfinv(0.5f), 0.47693627620446987f, 1e-5f);
}

TEST(TestNativeMath, TestCalcI0e) {
  EXPECT_NEAR(calc_i0e(0.0f), 1.0f, 1e-5f);
  EXPECT_NEAR(calc_i0e(1.0f), 0.46575960759364043f, 1e-5f);
}

TEST(TestNativeMath, TestCalcDigamma) {
  EXPECT_NEAR(calc_digamma(1.0f), -0.5772156649015329f, 1e-5f);
}

TEST(TestNativeMath, TestChbevl) {
  static const float coeff[] = {1.0f, 2.0f, 3.0f};
  EXPECT_NEAR(chbevl(0.0f, coeff, 3), 0.5f, 1e-6f);
}

TEST(TestNativeMath, TestExp2Impl) {
  EXPECT_FLOAT_EQ(exp2_impl(2.0f), 4.0f);
}

TEST(TestNativeMath, TestCalcI0) {
  EXPECT_NEAR(calc_i0(0.0f), 1.0f, 1e-5f);
}

TEST(TestNativeMath, TestCalcIgamma) {
  EXPECT_NEAR(calc_igamma(1.0f, 1.0f), 0.6321205588285577f, 1e-5f);
  EXPECT_NEAR(calc_igammac(1.0f, 1.0f), 0.36787944117144233f, 1e-5f);
}

TEST(TestNativeMath, TestPolevl) {
  static const float coeff[] = {1.0f, 1.0f, 1.0f};
  EXPECT_NEAR(polevl(1.0f, coeff, 2), 3.0f, 1e-6f);
}
