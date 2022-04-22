// Warning: this file is included twice in
// aten/src/ATen/test/cuda_complex_math_test.cu

#include <c10/util/complex.h>
#include <gtest/gtest.h>

#ifndef PI
#define PI 3.141592653589793238463
#endif

#ifndef tol
#define tol 1e-6
#endif

// Exponential functions

C10_DEFINE_TEST(TestExponential, IPi) {
  // exp(i*pi) = -1
  {
    c10::complex<float> e_i_pi = std::exp(c10::complex<float>(0, float(PI)));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    c10::complex<float> e_i_pi = ::exp(c10::complex<float>(0, float(PI)));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    c10::complex<double> e_i_pi = std::exp(c10::complex<double>(0, PI));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
  {
    c10::complex<double> e_i_pi = ::exp(c10::complex<double>(0, PI));
    C10_ASSERT_NEAR(e_i_pi.real(), -1, tol);
    C10_ASSERT_NEAR(e_i_pi.imag(), 0, tol);
  }
}

C10_DEFINE_TEST(TestExponential, EulerFormula) {
  // exp(ix) = cos(x) + i * sin(x)
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> e = std::exp(x);
    float expected_real = std::exp(x.real()) * std::cos(x.imag());
    float expected_imag = std::exp(x.real()) * std::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> e = ::exp(x);
    float expected_real = ::exp(x.real()) * ::cos(x.imag());
    float expected_imag = ::exp(x.real()) * ::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> e = std::exp(x);
    float expected_real = std::exp(x.real()) * std::cos(x.imag());
    float expected_imag = std::exp(x.real()) * std::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> e = ::exp(x);
    float expected_real = ::exp(x.real()) * ::cos(x.imag());
    float expected_imag = ::exp(x.real()) * ::sin(x.imag());
    C10_ASSERT_NEAR(e.real(), expected_real, tol);
    C10_ASSERT_NEAR(e.imag(), expected_imag, tol);
  }
}

C10_DEFINE_TEST(TestLog, Definition) {
  // log(x) = log(r) + i*theta
  {
    c10::complex<float> x(1.2, 3.4);
    c10::complex<float> l = std::log(x);
    float expected_real = std::log(std::abs(x));
    float expected_imag = std::arg(x);
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
  {
    c10::complex<float> x(1.2, 3.4);
    c10::complex<float> l = ::log(x);
    float expected_real = ::log(std::abs(x));
    float expected_imag = std::arg(x);
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(1.2, 3.4);
    c10::complex<double> l = std::log(x);
    float expected_real = std::log(std::abs(x));
    float expected_imag = std::arg(x);
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(1.2, 3.4);
    c10::complex<double> l = ::log(x);
    float expected_real = ::log(std::abs(x));
    float expected_imag = std::arg(x);
    C10_ASSERT_NEAR(l.real(), expected_real, tol);
    C10_ASSERT_NEAR(l.imag(), expected_imag, tol);
  }
}

C10_DEFINE_TEST(TestLog10, Rev) {
  // log10(10^x) = x
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> l = std::log10(std::pow(float(10), x));
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> l = ::log10(::pow(float(10), x));
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> l = std::log10(std::pow(double(10), x));
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> l = ::log10(::pow(double(10), x));
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
}

C10_DEFINE_TEST(TestLog2, Rev) {
  // log2(2^x) = x
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> l = std::log2(std::pow(float(2), x));
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> l = ::log2(std::pow(float(2), x));
    C10_ASSERT_NEAR(l.real(), float(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), float(1.2), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> l = std::log2(std::pow(double(2), x));
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> l = ::log2(std::pow(double(2), x));
    C10_ASSERT_NEAR(l.real(), double(0.1), tol);
    C10_ASSERT_NEAR(l.imag(), double(1.2), tol);
  }
}

// Power functions

C10_DEFINE_TEST(TestPowSqrt, Equal) {
  // x^0.5 = sqrt(x)
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::pow(x, float(0.5));
    c10::complex<float> z = std::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::pow(x, float(0.5));
    c10::complex<float> z = ::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::pow(x, double(0.5));
    c10::complex<double> z = std::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::pow(x, double(0.5));
    c10::complex<double> z = ::sqrt(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

C10_DEFINE_TEST(TestPow, Square) {
  // x^2 = x * x
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::pow(x, float(2));
    c10::complex<float> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::pow(x, float(2));
    c10::complex<float> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::pow(x, double(2));
    c10::complex<double> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::pow(x, double(2));
    c10::complex<double> z = x * x;
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

// Trigonometric functions and hyperbolic functions

C10_DEFINE_TEST(TestSinCosSinhCosh, Identity) {
  // sin(x + i * y) = sin(x) * cosh(y) + i * cos(x) * sinh(y)
  // cos(x + i * y) = cos(x) * cosh(y) - i * sin(x) * sinh(y)
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::sin(x);
    float expected_real = std::sin(x.real()) * std::cosh(x.imag());
    float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::sin(x);
    float expected_real = ::sin(x.real()) * ::cosh(x.imag());
    float expected_imag = ::cos(x.real()) * ::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::cos(x);
    float expected_real = std::cos(x.real()) * std::cosh(x.imag());
    float expected_imag = -std::sin(x.real()) * std::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::cos(x);
    float expected_real = ::cos(x.real()) * ::cosh(x.imag());
    float expected_imag = -::sin(x.real()) * ::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::sin(x);
    float expected_real = std::sin(x.real()) * std::cosh(x.imag());
    float expected_imag = std::cos(x.real()) * std::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::sin(x);
    float expected_real = ::sin(x.real()) * ::cosh(x.imag());
    float expected_imag = ::cos(x.real()) * ::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::cos(x);
    float expected_real = std::cos(x.real()) * std::cosh(x.imag());
    float expected_imag = -std::sin(x.real()) * std::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::cos(x);
    float expected_real = ::cos(x.real()) * ::cosh(x.imag());
    float expected_imag = -::sin(x.real()) * ::sinh(x.imag());
    C10_ASSERT_NEAR(y.real(), expected_real, tol);
    C10_ASSERT_NEAR(y.imag(), expected_imag, tol);
  }
}

C10_DEFINE_TEST(TestTan, Identity) {
  // tan(x) = sin(x) / cos(x)
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::tan(x);
    c10::complex<float> z = std::sin(x) / std::cos(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::tan(x);
    c10::complex<float> z = ::sin(x) / ::cos(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::tan(x);
    c10::complex<double> z = std::sin(x) / std::cos(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::tan(x);
    c10::complex<double> z = ::sin(x) / ::cos(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

C10_DEFINE_TEST(TestTanh, Identity) {
  // tanh(x) = sinh(x) / cosh(x)
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = std::tanh(x);
    c10::complex<float> z = std::sinh(x) / std::cosh(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<float> x(0.1, 1.2);
    c10::complex<float> y = ::tanh(x);
    c10::complex<float> z = ::sinh(x) / ::cosh(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = std::tanh(x);
    c10::complex<double> z = std::sinh(x) / std::cosh(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
  {
    c10::complex<double> x(0.1, 1.2);
    c10::complex<double> y = ::tanh(x);
    c10::complex<double> z = ::sinh(x) / ::cosh(x);
    C10_ASSERT_NEAR(y.real(), z.real(), tol);
    C10_ASSERT_NEAR(y.imag(), z.imag(), tol);
  }
}

// Rev trigonometric functions

C10_DEFINE_TEST(TestRevTrigonometric, Rev) {
  // asin(sin(x)) = x
  // acos(cos(x)) = x
  // atan(tan(x)) = x
  {
    c10::complex<float> x(0.5, 0.6);
    c10::complex<float> s = std::sin(x);
    c10::complex<float> ss = std::asin(s);
    c10::complex<float> c = std::cos(x);
    c10::complex<float> cc = std::acos(c);
    c10::complex<float> t = std::tan(x);
    c10::complex<float> tt = std::atan(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<float> x(0.5, 0.6);
    c10::complex<float> s = ::sin(x);
    c10::complex<float> ss = ::asin(s);
    c10::complex<float> c = ::cos(x);
    c10::complex<float> cc = ::acos(c);
    c10::complex<float> t = ::tan(x);
    c10::complex<float> tt = ::atan(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = std::sin(x);
    c10::complex<double> ss = std::asin(s);
    c10::complex<double> c = std::cos(x);
    c10::complex<double> cc = std::acos(c);
    c10::complex<double> t = std::tan(x);
    c10::complex<double> tt = std::atan(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = ::sin(x);
    c10::complex<double> ss = ::asin(s);
    c10::complex<double> c = ::cos(x);
    c10::complex<double> cc = ::acos(c);
    c10::complex<double> t = ::tan(x);
    c10::complex<double> tt = ::atan(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
}

// Rev hyperbolic functions

C10_DEFINE_TEST(TestRevHyperbolic, Rev) {
  // asinh(sinh(x)) = x
  // acosh(cosh(x)) = x
  // atanh(tanh(x)) = x
  {
    c10::complex<float> x(0.5, 0.6);
    c10::complex<float> s = std::sinh(x);
    c10::complex<float> ss = std::asinh(s);
    c10::complex<float> c = std::cosh(x);
    c10::complex<float> cc = std::acosh(c);
    c10::complex<float> t = std::tanh(x);
    c10::complex<float> tt = std::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<float> x(0.5, 0.6);
    c10::complex<float> s = ::sinh(x);
    c10::complex<float> ss = ::asinh(s);
    c10::complex<float> c = ::cosh(x);
    c10::complex<float> cc = ::acosh(c);
    c10::complex<float> t = ::tanh(x);
    c10::complex<float> tt = ::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = std::sinh(x);
    c10::complex<double> ss = std::asinh(s);
    c10::complex<double> c = std::cosh(x);
    c10::complex<double> cc = std::acosh(c);
    c10::complex<double> t = std::tanh(x);
    c10::complex<double> tt = std::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
  {
    c10::complex<double> x(0.5, 0.6);
    c10::complex<double> s = ::sinh(x);
    c10::complex<double> ss = ::asinh(s);
    c10::complex<double> c = ::cosh(x);
    c10::complex<double> cc = ::acosh(c);
    c10::complex<double> t = ::tanh(x);
    c10::complex<double> tt = ::atanh(t);
    C10_ASSERT_NEAR(x.real(), ss.real(), tol);
    C10_ASSERT_NEAR(x.imag(), ss.imag(), tol);
    C10_ASSERT_NEAR(x.real(), cc.real(), tol);
    C10_ASSERT_NEAR(x.imag(), cc.imag(), tol);
    C10_ASSERT_NEAR(x.real(), tt.real(), tol);
    C10_ASSERT_NEAR(x.imag(), tt.imag(), tol);
  }
}
