#pragma once
#include <ATen/native/IGammaCoefficients.h>
#include <c10/macros/Macros.h>
#include <cmath>
#include <cstdint>

namespace at::native::igamma_grad_detail {

// Note [igamma shape derivative]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Compute dQ(a, x)/da at fixed x; dP/da is its negative. The three regions
// differentiate the representations used by the forward incomplete gamma:
//
// * Lower series: DLMF 8.11.4, https://dlmf.nist.gov/8.11.E4. Write
//   P = B * sum(t_n), B = exp(a*log(x)-x-lgamma(a+1)), t_0 = 1,
//   t_n = t_(n-1)*x/(a+n). Then B'/B = beta = log(x)-psi(a+1),
//   and t_n'/t_n = -sum_{j=1}^n 1/(a+j).
// * Upper continued fraction: DLMF 8.9.2, https://dlmf.nist.gov/8.9.E2.
//   Q = a*B*h, so Q' = B*((1+a*beta)*h + a*h'). The modified Lentz
//   recurrence propagates h and h' together; it does not difference Q values.
// * Large-shape transition: differentiate the Temme expansion for Q in
//   DLMF 8.12.4/8.12.7, https://dlmf.nist.gov/8.12. The coefficients are
//   from the existing SciPy-derived forward implementation (see NOTICE).
//   Here lambda=x/a, eta=sign(lambda-1)*sqrt(2*(lambda-1-log(lambda))),
//   and d(eta)/da at fixed x is -(lambda-1)/(a*eta).
//
// The switches a>=20, abs((x-a)/a)<=0.35 select the tested transition region;
// they are accuracy choices, not a proof of a uniform error bound. Elsewhere
// x<a+1 selects the lower series. The scale uses Stirling corrections from
// a>=8, and the digamma recurrence shifts to x>=16 before its asymptotic
// expansion. Near lambda=1, power series avoid subtracting close logarithms.
//
// Compensated arithmetic is reserved for the small-result paths. The -699
// and -690 log thresholds enter them before exp reaches the normal/subnormal
// boundary (about -708). The 4096 bound guards integer range reduction, far
// outside the nonzero binary64 exponential range. final_scale rounds in units
// of 2^-1074, including ties to even; exponents below -1130 are negligible.
// The zero-envelope uses an action lower bound and an inflated prefactor
// (1500/rho + 1/(a*rho^2) on the lower side), with a log margin of 800
// beyond the prefactor. These conservative cutoffs have sampled validation;
// they do not establish an all-input accuracy guarantee.
//
// Series/fraction iterations are capped at 2000 with a 2e-15 convergence
// criterion. Failure to converge, a nonpositive transition bracket, or an
// out-of-range compensated scale returns NaN rather than an unchecked value.
struct Pair {
  double hi;
  double lo;
};
struct Uniform {
  double bracket;
  double action;
};
struct Sum {
  double value = 0.0;
  double correction = 0.0;
  C10_HOST_DEVICE void add(double x) {
    const double y = x - correction;
    const double next = value + y;
    correction = (next - value) - y;
    value = next;
  }
};

C10_HOST_DEVICE inline Pair two_sum(double a, double b) {
#if defined(__CUDA_ARCH__)
  // Error-free transforms require each indicated rounding even under --fmad=true.
  const double s = __dadd_rn(a, b);
  const double bb = __dsub_rn(s, a);
  return {s, __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb))};
#else
  const double s = a + b;
  const double bb = s - a;
  return {s, (a - (s - bb)) + (b - bb)};
#endif
}
C10_HOST_DEVICE inline Pair add(Pair a, Pair b) {
  const Pair s = two_sum(a.hi, b.hi);
  return two_sum(s.hi, s.lo + a.lo + b.lo);
}
C10_HOST_DEVICE inline Pair multiply(Pair a, Pair b) {
#if defined(__CUDA_ARCH__)
  const double p = __dmul_rn(a.hi, b.hi);
#else
  const double p = a.hi * b.hi;
#endif
  const double e = std::fma(a.hi, b.hi, -p) +
      a.hi * b.lo + a.lo * b.hi + a.lo * b.lo;
  return two_sum(p, e);
}
C10_HOST_DEVICE inline Pair divide(Pair a, double b) {
#if defined(__CUDA_ARCH__)
  const double q = __ddiv_rn(a.hi, b);
#else
  const double q = a.hi / b;
#endif
  const double r = (std::fma(-q, b, a.hi) + a.lo) / b;
  return two_sum(q, r);
}

C10_HOST_DEVICE inline Pair action_pair(double a, double x) {
  if (a == x) return {0.0, 0.0};
  int exponent;
  const double m = std::frexp(a, &exponent);
  const double dx = std::ldexp(x - a, -exponent);
  const Pair delta = divide({dx, 0.0}, m);
  Pair power{1.0, 0.0};
  Pair h{0.0, 0.0};
  for (int n = 0; n <= 64; ++n) {
    h = add(h, divide(multiply({2.0, 0.0}, power), n + 2.0));
    power = multiply(power, {-delta.hi, -delta.lo});
  }
  const Pair scaled = multiply(multiply({0.5 * dx, 0.0}, delta), h);
  return {std::ldexp(scaled.hi, exponent), std::ldexp(scaled.lo, exponent)};
}

C10_HOST_DEVICE inline int compare_scaled(double x, int exponent, double y) {
  if (x == 0.0) return y == 0.0 ? 0 : (y > 0.0 ? -1 : 1);
  if (y == 0.0) return x > 0.0 ? 1 : -1;
  if ((x < 0.0) != (y < 0.0)) return x < 0.0 ? -1 : 1;
  int x_exponent, y_exponent;
  const double mx = std::frexp(std::abs(x), &x_exponent);
  const double my = std::frexp(std::abs(y), &y_exponent);
  x_exponent += exponent;
  const int magnitude = x_exponent != y_exponent ?
      (x_exponent > y_exponent ? 1 : -1) : (mx == my ? 0 : (mx > my ? 1 : -1));
  return x < 0.0 ? -magnitude : magnitude;
}

C10_HOST_DEVICE inline double final_scale(Pair value, int exponent) {
  int normalization;
  const double hi = std::frexp(value.hi, &normalization);
  const int combined_exponent = exponent + normalization;
  if (combined_exponent > -1021)
    return std::ldexp(hi + std::ldexp(value.lo, -normalization), combined_exponent);
  if (combined_exponent < -1130) return 0.0;
  // Compare the unscaled low word with the midpoint gap. Scaling it first
  // could erase a nonzero residual that decides which side of a tie we are on.
  const int units_exponent = exponent + 1074;
  const double units_hi = std::ldexp(value.hi, units_exponent);
  const double integral = std::floor(units_hi);
  std::uint64_t n = static_cast<std::uint64_t>(integral);
  const double fraction = units_hi - integral;
  const int upper = compare_scaled(value.lo, units_exponent, 0.5 - fraction);
  const int lower = compare_scaled(value.lo, units_exponent, -0.5 - fraction);
  if (upper > 0 || (upper == 0 && (n & 1))) {
    ++n;
  } else if (lower < 0 || (lower == 0 && (n & 1))) {
    --n;
  }
  return std::ldexp(static_cast<double>(n), -1074);
}

C10_HOST_DEVICE inline double small_shape_tail_product(double a, double x, double factor) {
  const Pair logarithm = add({-x, 0.0}, {a * std::log(x) - std::lgamma(a + 1.0), 0.0});
  if (!std::isfinite(logarithm.hi) || std::abs(logarithm.hi) > 4096.0)
    return static_cast<double>(NAN);
  const int k = static_cast<int>(std::nearbyint(logarithm.hi * 1.4426950408889634074));
  const Pair reduction = add(logarithm, multiply({-static_cast<double>(k), 0.0},
      {0.69314718055994530942, 2.3190468138462995584e-17}));
  const double exponential = std::exp(reduction.hi);
  const Pair exp_pair = add({exponential, 0.0},
      multiply({exponential, 0.0}, {reduction.lo, 0.0}));
  return final_scale(multiply(exp_pair, {factor, 0.0}), k);
}

C10_HOST_DEVICE inline double compensated_product(double a, double x, double bracket) {
  const Pair s = action_pair(a, x);
  // The caller's conservative complete-derivative envelope handles huge actions.
  if (!std::isfinite(s.hi) || s.hi > 4096.0 || bracket <= 0.0)
    return static_cast<double>(NAN);
  const int k = static_cast<int>(std::nearbyint(-s.hi * 1.4426950408889634074));
  const Pair reduction = add(
      add({-s.hi, -s.lo}, multiply({-static_cast<double>(k), 0.0},
          {0.69314718055994530942, 2.3190468138462995584e-17})),
      {0.0, 0.0});
  const double exponential = std::exp(reduction.hi);
  const Pair exp_pair = add({exponential, 0.0},
      multiply({exponential, 0.0}, {reduction.lo, 0.0}));
  int e;
  double m = std::frexp(a, &e);
  if (e & 1) {
    m *= 2.0;
    --e;
  } else {
    m *= 4.0;
    e -= 2;
  }
  const double y = std::sqrt(m);
  const Pair root = two_sum(y, std::fma(-y, y, m) / (2.0 * y));
  const double q = 1.0 / root.hi;
  const Pair residual = add({1.0, 0.0}, multiply({-q, 0.0}, root));
  const Pair reciprocal = two_sum(q, (residual.hi + residual.lo) / root.hi);
  const Pair norm = multiply({0.3989422804014327, -2.49232720227773e-17}, reciprocal);
  return final_scale(multiply(multiply(exp_pair, norm), {bracket, 0.0}), k - e / 2);
}

C10_HOST_DEVICE inline double psi_positive(double x) {
  double correction = 0.0;
  while (x < 16.0) {
    correction -= 1.0 / x;
    x += 1.0;
  }
  const double r = 1.0 / x;
  const double z = r * r;
  const double polynomial = 1.0/12.0 + z * (-1.0/120.0 + z *
      (1.0/252.0 + z * (-1.0/240.0 + z * (1.0/132.0 + z *
      (-691.0/32760.0 + z * (1.0/12.0 - z * 3617.0/8160.0))))));
  return correction + std::log(x) - 0.5 * r - z * polynomial;
}

struct Scale {
  double log_b;
  double beta;
};
C10_HOST_DEVICE inline Scale scale_and_beta(double a, double x) {
  const double log_x = std::log(x);
  if (a < 8.0) return {a * log_x - x - std::lgamma(a + 1.0),
                       log_x - psi_positive(a + 1.0)};
  const double inv = 1.0 / a;
  const double z = inv * inv;
  const double gamma_correction = inv * (1.0/12.0 + z * (-1.0/360.0 + z *
      (1.0/1260.0 + z * (-1.0/1680.0 + z * (1.0/1188.0 + z *
      (-691.0/360360.0 + z * (1.0/156.0 - z * 3617.0/122400.0)))))));
  const double psi_delta = -0.5 * inv + z * (1.0/12.0 + z * (-1.0/120.0 + z *
      (1.0/252.0 + z * (-1.0/240.0 + z * (1.0/132.0 + z *
      (-691.0/32760.0 + z * (1.0/12.0 - z * 3617.0/8160.0)))))));
  const double delta = (x - a) / a;
  const double log_ratio = std::abs(delta) < 0.5 ? std::log1p(delta) : log_x - std::log(a);
  double centered;
  if (std::abs(delta) < 0.1) {
    Sum h;
    double power = 1.0;
    for (int j = 0; j < 30; ++j) {
      h.add(2.0 * power / (j + 2.0));
      power *= -delta;
    }
    centered = -0.5 * (x - a) * delta * h.value;
  } else {
    centered = a * log_ratio - x + a;
  }
  return {centered - 0.91893853320467274178 - 0.5 * std::log(a) - gamma_correction,
          log_ratio + psi_delta};
}

C10_HOST_DEVICE inline bool zero_envelope(double a, double x) {
  if (a < 8.0 || a == x) return false;
  const double delta = (x - a) / a;
  double action_bound, log_factor;
  if (delta > 0.0) {
    action_bound = 0.5 * (x - a) * (delta / (1.0 + delta));
    log_factor = std::log1p(a);
  } else {
    const double rho = -delta;
    action_bound = 0.5 * (a - x) * rho;
    const double t1 = std::log(1500.0) - std::log(rho);
    const double t2 = -std::log(a) - 2.0 * std::log(rho);
    const double m = t1 > t2 ? t1 : t2;
    log_factor = m + std::log(std::exp(t1 - m) + std::exp(t2 - m));
  }
  return action_bound > 800.0 + log_factor;
}

C10_HOST_DEVICE inline Uniform uniform_parts(double a, double x) {
  static constexpr double d[25][25] = ATEN_IGAMMA_ASYMPTOTIC_COEFFICIENTS;
  const double delta = (x - a) / a;
  double eta = 0.0, ratio = 1.0, leading = 1.0, log_lambda = 0.0;
  if (delta != 0.0) {
    log_lambda = std::log1p(delta);
    Sum sum;
    double term = 1.0;
    for (int j = 0; j < 50; ++j) {
      sum.add(2.0 * term / (j + 2.0));
      term *= -delta;
    }
    const double h = sum.value;
    const double root = std::sqrt(h);
    eta = delta * root;
    ratio = 1.0 / root;
    leading = (log_lambda / delta) * ratio;
  }
  Sum u, v;
  double power = 1.0;
  for (int k = 0; k < 25; ++k) {
    double c = d[k][24], derivative = 0.0;
    for (int n = 23; n >= 0; --n) {
      derivative = derivative * eta + c;
      c = c * eta + d[k][n];
    }
    u.add(c * power);
    v.add((k * c + ratio * derivative) * power);
    power /= a;
  }
  const double w = std::sqrt(a) * eta;
  return {leading + (log_lambda - 0.5 / a) * u.value - v.value / a, 0.5 * w * w};
}

C10_HOST_DEVICE inline double lower_series(double a, double x, Scale scale, int max_iterations = 2000) {
  double term = 1.0, harmonic = 0.0;
  Sum sum;
  sum.add(scale.beta);
  for (int n = 1; n <= max_iterations; ++n) {
    term *= x / (a + n);
    harmonic += 1.0 / (a + n);
    sum.add(term * (scale.beta - harmonic));
    const double r = x / (a + n + 1.0);
    const double tail = term * (r * std::abs(scale.beta - harmonic) / (1.0 - r) +
        r / ((a + n + 1.0) * (1.0 - r) * (1.0 - r)));
    if (tail <= 2e-15 * std::abs(sum.value)) {
      if (sum.value >= 0.0) break;
      return std::exp(scale.log_b + std::log(-sum.value));
    }
  }
  return static_cast<double>(NAN);
}

C10_HOST_DEVICE inline double upper_fraction(double a, double x, Scale scale, int max_iterations = 2000) {
  double denominator = (x - a) + 1.0;
  double numerator_ratio = 1e300;
  double numerator_derivative = 0.0;
  double inverse_denominator = 1.0 / denominator;
  double inverse_derivative = (1.0 / denominator) / denominator;
  double fraction = inverse_denominator;
  double fraction_derivative = inverse_derivative;
  double previous = 0.0;
  int stable = 0;
  for (int i = 1; i <= max_iterations; ++i) {
    const double coefficient = i * (a - i);
    denominator += 2.0;
    const double next_denominator = denominator + coefficient * inverse_denominator;
    const double next_denominator_derivative =
        -1.0 + i * inverse_denominator + coefficient * inverse_derivative;
    const double next_numerator = denominator + coefficient / numerator_ratio;
    const double next_numerator_derivative = -1.0 + i / numerator_ratio -
        coefficient * (numerator_derivative / numerator_ratio) / numerator_ratio;
    inverse_denominator = 1.0 / next_denominator;
    inverse_derivative =
        -next_denominator_derivative * inverse_denominator * inverse_denominator;
    numerator_ratio = next_numerator;
    numerator_derivative = next_numerator_derivative;
    const double delta = numerator_ratio * inverse_denominator;
    const double delta_derivative = numerator_derivative * inverse_denominator +
        numerator_ratio * inverse_derivative;
    fraction_derivative = fraction_derivative * delta + fraction * delta_derivative;
    fraction *= delta;
    const double factor = (1.0 + a * scale.beta) * fraction + a * fraction_derivative;
    if (i > 1 && factor > 0.0 && std::abs(factor - previous) <= 2e-15 * factor &&
        std::abs(delta - 1.0) <= 2e-15) {
      if (++stable >= 3) {
        if (a < 8.0 && scale.log_b < -690.0) return small_shape_tail_product(a, x, factor);
        return std::exp(scale.log_b + std::log(factor));
      }
    } else {
      stable = 0;
    }
    previous = factor;
  }
  return static_cast<double>(NAN);
}

C10_HOST_DEVICE inline double derivative_q(double a, double x) {
  if (std::isnan(a) || std::isnan(x) || a <= 0.0 || x < 0.0)
    return static_cast<double>(NAN);
  if (std::isinf(a)) return std::isinf(x) ? static_cast<double>(NAN) : 0.0;
  if (x == 0.0 || std::isinf(x)) return 0.0;
  if (zero_envelope(a, x)) return 0.0;
  const double delta = (x - a) / a;
  if (a >= 20.0 && std::abs(delta) <= 0.35) {
    const Uniform result = uniform_parts(a, x);
    if (result.bracket <= 0.0) return static_cast<double>(NAN);
    const double log_g = -result.action - 0.5 * std::log(a) -
        0.91893853320467274178 + std::log(result.bracket);
    return log_g < -699.0 ? compensated_product(a, x, result.bracket) : std::exp(log_g);
  }
  const Scale scale = scale_and_beta(a, x);
  if (x < a + 1.0) return lower_series(a, x, scale);
  if (scale.log_b + std::log1p(a * (std::abs(scale.beta) + 1.0)) < -800.0) return 0.0;
  return upper_fraction(a, x, scale);
}
} // namespace at::native::igamma_grad_detail
