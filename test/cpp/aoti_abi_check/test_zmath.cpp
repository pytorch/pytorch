#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/zmath.h>

TEST(TestZmath, TestZabsComplex) {
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::zabs<torch::headeronly::complex<float>, float>(
          torch::headeronly::complex<float>(3.0f, 4.0f)),
      5.0f);
  EXPECT_DOUBLE_EQ(
      torch::headeronly::native::
          zabs<torch::headeronly::complex<double>, double>(
              torch::headeronly::complex<double>(3.0, 4.0)),
      5.0);
}

TEST(TestZmath, TestAngleImpl) {
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::angle_impl(-1.0f),
      static_cast<float>(torch::headeronly::pi<double>));
  EXPECT_FLOAT_EQ(torch::headeronly::native::angle_impl(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::angle_impl(
          torch::headeronly::complex<float>(0.0f, 1.0f)),
      static_cast<float>(torch::headeronly::pi<double> / 2.0));
}

TEST(TestZmath, TestRealImagImpl) {
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::real_impl(
          torch::headeronly::complex<float>(1.5f, 2.5f)),
      1.5f);
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::imag_impl(
          torch::headeronly::complex<float>(1.5f, 2.5f)),
      2.5f);
}

TEST(TestZmath, TestMinMaxImpl) {
  EXPECT_FLOAT_EQ(torch::headeronly::native::max_impl(1.0f, 2.0f), 2.0f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::min_impl(1.0f, 2.0f), 1.0f);
}

TEST(TestZmath, TestConjImpl) {
  auto z = torch::headeronly::native::conj_impl(
      torch::headeronly::complex<float>(1.0f, 2.0f));
  EXPECT_FLOAT_EQ(z.real(), 1.0f);
  EXPECT_FLOAT_EQ(z.imag(), -2.0f);
}
