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

TEST(TestPowSqrt, Equal) {
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

TEST(TestPow, Square) {
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

// Trigonometric functions and hyperbolic functions

TEST(TestSinCosSinhCosh, Identity) {
  // sin(x + i * y) = sin(x) * cosh(y) + i * cos(x) * sinh(y)
  // cos(x + i * y) = cos(x) * cosh(y) - i * sin(x) * sinh(y)
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::sin(x);
  float expected_real = std::sin(x.real()) * std::cosh(x.imag());
  float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
  ASSERT_NEAR(y.real(), expected_real, tol);
  ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::cos(x);
  float expected_real = std::cos(x.real()) * std::cosh(x.imag());
  float expected_imag = - std::sin(x.real()) * std::sinh(x.imag());
  ASSERT_NEAR(y.real(), expected_real, tol);
  ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::sin(x);
  float expected_real = std::sin(x.real()) * std::cosh(x.imag());
  float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
  ASSERT_NEAR(y.real(), expected_real, tol);
  ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::cos(x);
  float expected_real = std::cos(x.real()) * std::cosh(x.imag());
  float expected_imag = - std::sin(x.real()) * std::sinh(x.imag());
  ASSERT_NEAR(y.real(), expected_real, tol);
  ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
}

TEST(TestTan, Identity) {
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::tan(x);
  c10::complex<float> z = std::sin(x) / std::cos(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::tan(x);
  c10::complex<double> z = std::sin(x) / std::cos(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

TEST(TestTanh, Identity) {
  {
  c10::complex<float> x(0.1, 1.2);
  c10::complex<float> y = std::tanh(x);
  c10::complex<float> z = std::sinh(x) / std::cosh(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
  c10::complex<double> x(0.1, 1.2);
  c10::complex<double> y = std::tanh(x);
  c10::complex<double> z = std::sinh(x) / std::cosh(x);
  ASSERT_NEAR(y.real(), z.real(), tol);
  ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

// Rev trigonometric functions

TEST(TestRevTrigonometric, Rev) {
  {
  c10::complex<float> x(0.5, 0.6);
  c10::complex<float> s = std::sin(x);
  c10::complex<float> ss = std::asin(s);
  c10::complex<float> c = std::cos(x);
  c10::complex<float> cc = std::acos(c);
  c10::complex<float> t = std::tan(x);
  c10::complex<float> tt = std::atan(t);
  ASSERT_NEAR(x.real(), ss.real(), tol);
  ASSERT_NEAR(x.imag(), ss.imag(), tol);
  ASSERT_NEAR(x.real(), cc.real(), tol);
  ASSERT_NEAR(x.imag(), cc.imag(), tol);
  ASSERT_NEAR(x.real(), tt.real(), tol);
  ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
  c10::complex<double> x(0.5, 0.6);
  c10::complex<double> s = std::sin(x);
  c10::complex<double> ss = std::asin(s);
  c10::complex<double> c = std::cos(x);
  c10::complex<double> cc = std::acos(c);
  c10::complex<double> t = std::tan(x);
  c10::complex<double> tt = std::atan(t);
  ASSERT_NEAR(x.real(), ss.real(), tol);
  ASSERT_NEAR(x.imag(), ss.imag(), tol);
  ASSERT_NEAR(x.real(), cc.real(), tol);
  ASSERT_NEAR(x.imag(), cc.imag(), tol);
  ASSERT_NEAR(x.real(), tt.real(), tol);
  ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
}

// Rev hyperbolic functions

TEST(TestRevHyperbolic, Rev) {
  {
  c10::complex<float> x(0.5, 0.6);
  c10::complex<float> s = std::sinh(x);
  c10::complex<float> ss = std::asinh(s);
  c10::complex<float> c = std::cosh(x);
  c10::complex<float> cc = std::acosh(c);
  c10::complex<float> t = std::tanh(x);
  c10::complex<float> tt = std::atanh(t);
  ASSERT_NEAR(x.real(), ss.real(), tol);
  ASSERT_NEAR(x.imag(), ss.imag(), tol);
  ASSERT_NEAR(x.real(), cc.real(), tol);
  ASSERT_NEAR(x.imag(), cc.imag(), tol);
  ASSERT_NEAR(x.real(), tt.real(), tol);
  ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
  c10::complex<double> x(0.5, 0.6);
  c10::complex<double> s = std::sinh(x);
  c10::complex<double> ss = std::asinh(s);
  c10::complex<double> c = std::cosh(x);
  c10::complex<double> cc = std::acosh(c);
  c10::complex<double> t = std::tanh(x);
  c10::complex<double> tt = std::atanh(t);
  ASSERT_NEAR(x.real(), ss.real(), tol);
  ASSERT_NEAR(x.imag(), ss.imag(), tol);
  ASSERT_NEAR(x.real(), cc.real(), tol);
  ASSERT_NEAR(x.imag(), cc.imag(), tol);
  ASSERT_NEAR(x.real(), tt.real(), tol);
  ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
}
