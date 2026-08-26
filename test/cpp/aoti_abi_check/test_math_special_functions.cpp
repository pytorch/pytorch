#include <gtest/gtest.h>

#include <torch/headeronly/cpu/vec/zmath.h>
#include <torch/headeronly/util/Math.h>
#include <cmath>

namespace torch {
namespace aot_inductor {

// torch::headeronly::native (torch/headeronly/util/Math.h) -- special
// functions used by ATen/native/Math.h. Internal Cephes-derived helpers
// (chbevl, polevl, ratevl, abs_impl, getHermitianLimit,
// chebyshev_coefficients_i0e_A/B, chebyshev_coefficients_i1e_A/B,
// lanczos_sum_expg_scaled, erfcx_y100, and the _igam(c)_helper_* family) are
// exercised transitively through the public functions tested below.

TEST(TestMathSpecialFunctions, TestErfinvAndGamma) {
  using namespace torch::headeronly::native;

  EXPECT_NEAR(calc_erfinv(0.0), 0.0, 1e-6);
  EXPECT_NEAR(calc_erfinv(0.5), 0.4769362762, 1e-6);

  // zeta(2, 1) == pi^2 / 6
  EXPECT_NEAR((zeta<double, false>(2.0, 1.0)), 1.6449340668, 1e-6);

  EXPECT_NEAR(trigamma(1.0), 1.6449340668, 1e-4);
  EXPECT_NEAR(trigamma(1.0f), 1.6449340668f, 1e-3f);

  // digamma(1) == -euler_gamma
  EXPECT_NEAR(calc_digamma(1.0), -0.5772156649, 1e-6);

  // polygamma(2, 1) == -2 * zeta(3, 1) == -2 * Apery's constant
  EXPECT_NEAR(calc_polygamma(1.0, 2), -2.4041138063, 1e-6);

  EXPECT_GT(calc_igamma(1.0, 1.0), 0.0);
  EXPECT_GT(calc_igammac(1.0, 1.0), 0.0);
  EXPECT_NEAR(calc_igamma(1.0, 1.0) + calc_igammac(1.0, 1.0), 1.0, 1e-9);
}

TEST(TestMathSpecialFunctions, TestGcdAndExp2) {
  using namespace torch::headeronly::native;

  EXPECT_EQ(calc_gcd(12, 18), 6);
  EXPECT_EQ(calc_gcd(-12, 18), 6);

  EXPECT_NEAR(exp2_impl(3.0), 8.0, 1e-9);
  auto c = exp2_impl(torch::headeronly::complex<double>(1.0, 0.0));
  EXPECT_NEAR(c.real(), 2.0, 1e-9);
}

TEST(TestMathSpecialFunctions, TestBesselI) {
  using namespace torch::headeronly::native;

  EXPECT_NEAR(calc_i0(0.0), 1.0, 1e-9);
  EXPECT_NEAR(calc_i0e(0.0), 1.0, 1e-9);
  EXPECT_NEAR(calc_i1(0.0), 0.0, 1e-9);
  EXPECT_NEAR(calc_i1e(0.0), 0.0, 1e-9);
}

TEST(TestMathSpecialFunctions, TestNdtriAndErfcx) {
  using namespace torch::headeronly::native;

  EXPECT_NEAR(calc_ndtri(0.5), 0.0, 1e-9);
  EXPECT_LT(calc_log_ndtr(0.0), 0.0);
  EXPECT_NEAR(calc_erfcx(0.0), 1.0, 1e-9);
}

TEST(TestMathSpecialFunctions, TestAiryAndBessel) {
  using namespace torch::headeronly::native;

  EXPECT_FALSE(std::isnan(airy_ai_forward(0.5)));
  EXPECT_NEAR(bessel_j0_forward(0.0), 1.0, 1e-9);
  EXPECT_NEAR(bessel_j1_forward(0.0), 0.0, 1e-9);
  EXPECT_FALSE(std::isnan(bessel_y0_forward(1.0)));
  EXPECT_FALSE(std::isnan(bessel_y1_forward(1.0)));
}

TEST(TestMathSpecialFunctions, TestOrthogonalPolynomials) {
  using namespace torch::headeronly::native;

  // All of these T_n(x)-style polynomials equal 1 at n == 0.
  EXPECT_NEAR(chebyshev_polynomial_t_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(chebyshev_polynomial_u_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(chebyshev_polynomial_v_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(chebyshev_polynomial_w_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(
      chebyshev_polynomial_t_forward(0.3, static_cast<double>(0)), 1.0, 1e-9);

  EXPECT_NEAR(hermite_polynomial_h_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(hermite_polynomial_he_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(
      hermite_polynomial_h_forward(0.3, static_cast<double>(0)), 1.0, 1e-9);

  EXPECT_NEAR(laguerre_polynomial_l_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(legendre_polynomial_p_forward(0.3, int64_t{0}), 1.0, 1e-9);

  EXPECT_NEAR(
      shifted_chebyshev_polynomial_t_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(
      shifted_chebyshev_polynomial_u_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(
      shifted_chebyshev_polynomial_v_forward(0.3, int64_t{0}), 1.0, 1e-9);
  EXPECT_NEAR(
      shifted_chebyshev_polynomial_w_forward(0.3, int64_t{0}), 1.0, 1e-9);
}

TEST(TestMathSpecialFunctions, TestModifiedBesselAndSpherical) {
  using namespace torch::headeronly::native;

  EXPECT_NEAR(modified_bessel_i0_forward(0.0), 1.0, 1e-9);
  EXPECT_NEAR(modified_bessel_i1_forward(0.0), 0.0, 1e-9);
  EXPECT_FALSE(std::isnan(modified_bessel_k0_forward(1.0)));
  EXPECT_FALSE(std::isnan(modified_bessel_k1_forward(1.0)));
  EXPECT_FALSE(std::isnan(scaled_modified_bessel_k0_forward(1.0)));
  EXPECT_FALSE(std::isnan(scaled_modified_bessel_k1_forward(1.0)));

  // spherical_bessel_j0(x) == sin(x) / x
  EXPECT_NEAR(spherical_bessel_j0_forward(0.3), std::sin(0.3) / 0.3, 1e-9);
}

// torch::headeronly::native::CPU_CAPABILITY (torch/headeronly/cpu/vec/zmath.h)
TEST(TestMathSpecialFunctions, TestZmath) {
  using namespace torch::headeronly::native;
  using torch::headeronly::complex;

  EXPECT_NEAR(zabs(complex<double>(3.0, 4.0)).real(), 5.0, 1e-9);
  EXPECT_NEAR(
      (zabs<complex<double>, double>(complex<double>(3.0, 4.0))), 5.0, 1e-9);

  EXPECT_NEAR(angle_impl(-1.0), torch::headeronly::pi<double>, 1e-9);
  EXPECT_NEAR(angle_impl(1.0), 0.0, 1e-9);

  EXPECT_NEAR(real_impl(complex<double>(1.0, 2.0)).real(), 1.0, 1e-9);
  EXPECT_NEAR(imag_impl(complex<double>(1.0, 2.0)).real(), 2.0, 1e-9);

  auto c = conj_impl(complex<double>(1.0, 2.0));
  EXPECT_NEAR(c.real(), 1.0, 1e-9);
  EXPECT_NEAR(c.imag(), -2.0, 1e-9);

  EXPECT_NEAR(ceil_impl(1.2), 2.0, 1e-9);
  EXPECT_NEAR(floor_impl(1.8), 1.0, 1e-9);
  EXPECT_NEAR(round_impl(1.5), 2.0, 1e-9);
  EXPECT_NEAR(trunc_impl(1.8), 1.0, 1e-9);

  auto s = sgn_impl(complex<double>(3.0, 4.0));
  EXPECT_NEAR(s.real(), 0.6, 1e-9);
  EXPECT_NEAR(s.imag(), 0.8, 1e-9);

  EXPECT_EQ(max_impl(1.0, 2.0), 2.0);
  EXPECT_EQ(min_impl(1.0, 2.0), 1.0);
}

} // namespace aot_inductor
} // namespace torch
