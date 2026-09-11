#include <gtest/gtest.h>

#include <torch/headeronly/native/cpu/zmath.h>

TEST(TestZmath, TestZabsComplex) {
  EXPECT_FLOAT_EQ(
      (torch::headeronly::native::
           zabs<torch::headeronly::complex<float>, float>(
               torch::headeronly::complex<float>(3.0f, 4.0f))),
      5.0f);
  EXPECT_DOUBLE_EQ(
      (torch::headeronly::native::
           zabs<torch::headeronly::complex<double>, double>(
               torch::headeronly::complex<double>(3.0, 4.0))),
      5.0);
}

TEST(TestZmath, TestAngleImpl) {
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::angle_impl(-1.0f),
      static_cast<float>(torch::headeronly::pi<double>));
  EXPECT_FLOAT_EQ(torch::headeronly::native::angle_impl(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(
      (torch::headeronly::native::
           angle_impl<torch::headeronly::complex<float>, float>(
               torch::headeronly::complex<float>(0.0f, 1.0f))),
      static_cast<float>(torch::headeronly::pi<double> / 2.0));
}

TEST(TestZmath, TestRealImagImpl) {
  EXPECT_FLOAT_EQ(
      (torch::headeronly::native::
           real_impl<torch::headeronly::complex<float>, float>(
               torch::headeronly::complex<float>(1.5f, 2.5f))),
      1.5f);
  EXPECT_FLOAT_EQ(
      (torch::headeronly::native::
           imag_impl<torch::headeronly::complex<float>, float>(
               torch::headeronly::complex<float>(1.5f, 2.5f))),
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

TEST(TestZmath, TestRoundingImpls) {
  using Complex = torch::headeronly::complex<float>;

  const auto ceil = torch::headeronly::native::ceil_impl(Complex(1.2f, -2.8f));
  EXPECT_FLOAT_EQ(ceil.real(), 2.0f);
  EXPECT_FLOAT_EQ(ceil.imag(), -2.0f);

  const auto floor =
      torch::headeronly::native::floor_impl(Complex(1.2f, -2.8f));
  EXPECT_FLOAT_EQ(floor.real(), 1.0f);
  EXPECT_FLOAT_EQ(floor.imag(), -3.0f);

  const auto round =
      torch::headeronly::native::round_impl(Complex(1.6f, -2.4f));
  EXPECT_FLOAT_EQ(round.real(), 2.0f);
  EXPECT_FLOAT_EQ(round.imag(), -2.0f);

  const auto trunc =
      torch::headeronly::native::trunc_impl(Complex(1.8f, -2.8f));
  EXPECT_FLOAT_EQ(trunc.real(), 1.0f);
  EXPECT_FLOAT_EQ(trunc.imag(), -2.0f);
}

TEST(TestZmath, TestSgnImpl) {
  using Complex = torch::headeronly::complex<float>;
  const auto sign = torch::headeronly::native::sgn_impl(Complex(3.0f, 4.0f));
  EXPECT_FLOAT_EQ(sign.real(), 0.6f);
  EXPECT_FLOAT_EQ(sign.imag(), 0.8f);
}
