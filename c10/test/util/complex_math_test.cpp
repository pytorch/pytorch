#include <c10/util/complex.h>
#include <gtest/gtest.h>

const double PI = 3.141592653589793238463;
const double tol = 1e-6;

// Exponential functions

TEST(TestExponential, IPi) {
  // exp(i*pi) = -1
  {
  c10::complex<float> e_i_pi = std::exp(c10::complex<float>(0, float(PI)));
  ASSERT_NEAR(e_i_pi.real(), -1, tol);
  ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
  c10::complex<double> e_i_pi = std::exp(c10::complex<double>(0, PI));
  ASSERT_NEAR(e_i_pi.real(), -1, tol);
  ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
}

TEST(TestExponential, EulerFormula) {
  // exp(ix) = cos(x) + i * sin(x)
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> e = std::exp(x);
  float expected_real = std::exp(x.real()) * std::cos(x.imag());
  float expected_imag = std::exp(x.real()) * std::sin(x.imag());
  ASSERT_NEAR(e.real(), expected_real, tol);
  ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> e = std::exp(x);
  float expected_real = std::exp(x.real()) * std::cos(x.imag());
  float expected_imag = std::exp(x.real()) * std::sin(x.imag());
  ASSERT_NEAR(e.real(), expected_real, tol);
  ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
}

TEST(TestLog, Definition) {
  // log(x) = log(r) + i*theta
  {
  c10::complex<float> x(1.2, 3.4);
  c10::complex<float> l = std::log(x);
  float expected_real = std::log(std::abs(x));
  float expected_imag = std::arg(x);
  ASSERT_NEAR(l.real(), expected_real, tol);
  ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
  {
  c10::complex<double> x(1.2, 3.4);
  c10::complex<double> l = std::log(x);
  float expected_real = std::log(std::abs(x));
  float expected_imag = std::arg(x);
  ASSERT_NEAR(l.real(), expected_real, tol);
  ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
}

TEST(TestLog10, Rev) {
  // log10(10^x) = x
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> l = std::log10(std::pow(float(10), x));
  ASSERT_NEAR(l.real(), float(0.1), tol);
  ASSERT_NEAR(l.imag(), float(1.2), tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> l = std::log10(std::pow(double(10), x));
  ASSERT_NEAR(l.real(), double(0.1), tol);
  ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
}

// Power functions

TEST(TestPowSqrt, equal) {
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::pow(x, float(0.5));
  c10::complex<float> z = std::sqrt(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::pow(x, double(0.5));
  c10::complex<double> z = std::sqrt(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

TEST(TestPow, square) {
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::pow(x, float(2));
  c10::complex<float> z = x * x;
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::pow(x, double(2));
  c10::complex<double> z = x * x;
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}
