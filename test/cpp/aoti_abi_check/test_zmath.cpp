#include <gtest/gtest.h>

#include <torch/headeronly/native/cpu/zmath.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

template <typename T>
void ExpectComplexEq(
    const torch::headeronly::complex<T>& actual,
    const torch::headeronly::complex<T>& expected) {
  if constexpr (std::is_same_v<T, double>) {
    EXPECT_DOUBLE_EQ(actual.real(), expected.real());
    EXPECT_DOUBLE_EQ(actual.imag(), expected.imag());
  } else {
    EXPECT_FLOAT_EQ(actual.real(), expected.real());
    EXPECT_FLOAT_EQ(actual.imag(), expected.imag());
  }
}

} // namespace

TEST(TestZmath, TestZabsComplex) {
  EXPECT_FLOAT_EQ(torch::headeronly::native::zabs(-3.0f), -3.0f);
  ExpectComplexEq(
      torch::headeronly::native::zabs(
          torch::headeronly::complex<float>(3.0f, 4.0f)),
      torch::headeronly::complex<float>(5.0f, 0.0f));
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
  ExpectComplexEq(
      torch::headeronly::native::zabs(
          torch::headeronly::complex<double>(3.0, 4.0)),
      torch::headeronly::complex<double>(5.0, 0.0));
}

TEST(TestZmath, TestAngleImpl) {
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::angle_impl(-1.0f),
      static_cast<float>(torch::headeronly::pi<double>));
  EXPECT_FLOAT_EQ(torch::headeronly::native::angle_impl(1.0f), 0.0f);
  EXPECT_FLOAT_EQ(
      torch::headeronly::native::angle_impl(-0.0f),
      static_cast<float>(torch::headeronly::pi<double>));
  EXPECT_DOUBLE_EQ(
      torch::headeronly::native::angle_impl(-0.0),
      torch::headeronly::pi<double>);
  EXPECT_NEAR(
      static_cast<float>(torch::headeronly::native::angle_impl(
          torch::headeronly::Half{-0.0f})),
      static_cast<float>(torch::headeronly::pi<double>),
      2e-3f);
  EXPECT_NEAR(
      static_cast<float>(torch::headeronly::native::angle_impl(
          torch::headeronly::BFloat16{-0.0f})),
      static_cast<float>(torch::headeronly::pi<double>),
      2e-3f);
  EXPECT_TRUE(std::isnan(torch::headeronly::native::angle_impl(
      std::numeric_limits<float>::quiet_NaN())));

  ExpectComplexEq(
      torch::headeronly::native::angle_impl(
          torch::headeronly::complex<float>(0.0f, 1.0f)),
      torch::headeronly::complex<float>(
          static_cast<float>(torch::headeronly::pi<double> / 2.0), 0.0f));
  EXPECT_FLOAT_EQ(
      (torch::headeronly::native::
           angle_impl<torch::headeronly::complex<float>, float>(
               torch::headeronly::complex<float>(0.0f, 1.0f))),
      static_cast<float>(torch::headeronly::pi<double> / 2.0));

  ExpectComplexEq(
      torch::headeronly::native::angle_impl(
          torch::headeronly::complex<double>(0.0, 1.0)),
      torch::headeronly::complex<double>(
          torch::headeronly::pi<double> / 2.0, 0.0));
  EXPECT_DOUBLE_EQ(
      (torch::headeronly::native::
           angle_impl<torch::headeronly::complex<double>, double>(
               torch::headeronly::complex<double>(0.0, 1.0))),
      torch::headeronly::pi<double> / 2.0);
}

TEST(TestZmath, TestRealImagImpl) {
  EXPECT_FLOAT_EQ(torch::headeronly::native::real_impl(1.5f), 1.5f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::imag_impl(1.5f), 0.0f);

  ExpectComplexEq(
      torch::headeronly::native::real_impl(
          torch::headeronly::complex<float>(1.5f, 2.5f)),
      torch::headeronly::complex<float>(1.5f, 0.0f));
  ExpectComplexEq(
      torch::headeronly::native::imag_impl(
          torch::headeronly::complex<float>(1.5f, 2.5f)),
      torch::headeronly::complex<float>(2.5f, 0.0f));
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

  ExpectComplexEq(
      torch::headeronly::native::real_impl(
          torch::headeronly::complex<double>(1.5, 2.5)),
      torch::headeronly::complex<double>(1.5, 0.0));
  ExpectComplexEq(
      torch::headeronly::native::imag_impl(
          torch::headeronly::complex<double>(1.5, 2.5)),
      torch::headeronly::complex<double>(2.5, 0.0));
  EXPECT_DOUBLE_EQ(
      (torch::headeronly::native::
           real_impl<torch::headeronly::complex<double>, double>(
               torch::headeronly::complex<double>(1.5, 2.5))),
      1.5);
  EXPECT_DOUBLE_EQ(
      (torch::headeronly::native::
           imag_impl<torch::headeronly::complex<double>, double>(
               torch::headeronly::complex<double>(1.5, 2.5))),
      2.5);
}

TEST(TestZmath, TestMinMaxImpl) {
  EXPECT_FLOAT_EQ(torch::headeronly::native::max_impl(1.0f, 2.0f), 2.0f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::min_impl(1.0f, 2.0f), 1.0f);
  using Complex = torch::headeronly::complex<float>;
  EXPECT_EQ(
      torch::headeronly::native::max_impl(
          Complex(3.0f, 4.0f), Complex(1.0f, 1.0f)),
      Complex(3.0f, 4.0f));
  EXPECT_EQ(
      torch::headeronly::native::min_impl(
          Complex(3.0f, 4.0f), Complex(1.0f, 1.0f)),
      Complex(1.0f, 1.0f));
}

TEST(TestZmath, TestConjImpl) {
  EXPECT_FLOAT_EQ(torch::headeronly::native::conj_impl(2.0f), 2.0f);
  using ComplexHalf = torch::headeronly::complex<torch::headeronly::Half>;
  ExpectComplexEq(
      torch::headeronly::native::conj_impl(ComplexHalf(
          torch::headeronly::Half{1.0f}, torch::headeronly::Half{2.0f})),
      ComplexHalf(
          torch::headeronly::Half{1.0f}, torch::headeronly::Half{-2.0f}));
  ExpectComplexEq(
      torch::headeronly::native::conj_impl(
          torch::headeronly::complex<float>(1.0f, 2.0f)),
      torch::headeronly::complex<float>(1.0f, -2.0f));
  ExpectComplexEq(
      torch::headeronly::native::conj_impl(
          torch::headeronly::complex<double>(1.0, 2.0)),
      torch::headeronly::complex<double>(1.0, -2.0));
}

TEST(TestZmath, TestRoundingImpls) {
  using Complex = torch::headeronly::complex<float>;

  EXPECT_FLOAT_EQ(torch::headeronly::native::ceil_impl(1.2f), 2.0f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::floor_impl(1.2f), 1.0f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::round_impl(1.6f), 2.0f);
  EXPECT_FLOAT_EQ(torch::headeronly::native::trunc_impl(1.8f), 1.0f);

  ExpectComplexEq(
      torch::headeronly::native::ceil_impl(Complex(1.2f, -2.8f)),
      Complex(2.0f, -2.0f));
  ExpectComplexEq(
      torch::headeronly::native::floor_impl(Complex(1.2f, -2.8f)),
      Complex(1.0f, -3.0f));
  ExpectComplexEq(
      torch::headeronly::native::round_impl(Complex(1.6f, -2.4f)),
      Complex(2.0f, -2.0f));
  ExpectComplexEq(
      torch::headeronly::native::trunc_impl(Complex(1.8f, -2.8f)),
      Complex(1.0f, -2.0f));

  using ComplexDouble = torch::headeronly::complex<double>;
  ExpectComplexEq(
      torch::headeronly::native::ceil_impl(ComplexDouble(1.2, -2.8)),
      ComplexDouble(2.0, -2.0));
  ExpectComplexEq(
      torch::headeronly::native::floor_impl(ComplexDouble(1.2, -2.8)),
      ComplexDouble(1.0, -3.0));
  ExpectComplexEq(
      torch::headeronly::native::round_impl(ComplexDouble(1.6, -2.4)),
      ComplexDouble(2.0, -2.0));
  ExpectComplexEq(
      torch::headeronly::native::trunc_impl(ComplexDouble(1.8, -2.8)),
      ComplexDouble(1.0, -2.0));
}

TEST(TestZmath, TestSgnImpl) {
  using Complex = torch::headeronly::complex<float>;
  ExpectComplexEq(
      torch::headeronly::native::sgn_impl(Complex(3.0f, 4.0f)),
      Complex(0.6f, 0.8f));
  ExpectComplexEq(
      torch::headeronly::native::sgn_impl(Complex(0.0f, 0.0f)),
      Complex(0.0f, 0.0f));
  using ComplexDouble = torch::headeronly::complex<double>;
  ExpectComplexEq(
      torch::headeronly::native::sgn_impl(ComplexDouble(3.0, 4.0)),
      ComplexDouble(0.6, 0.8));
}
