#pragma once

#include <ATen/native/special_functions/polar_pi.h>
#include <ATen/native/special_functions/detail/euler_maclaurin.h>
#include <ATen/native/special_functions/detail/factorial.h>
#include <ATen/native/special_functions/detail/is_equal.h>
#include <ATen/native/special_functions/detail/is_imag.h>
#include <ATen/native/special_functions/detail/is_integer.h>
#include <ATen/native/special_functions/detail/is_real.h>
#include <ATen/native/special_functions/detail/is_zero.h>
#include <ATen/native/special_functions/detail/numeric_t.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/reciprocal_gamma.h>
#include <ATen/native/special_functions/detail/riemann_zeta.h>
#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
class Terminator {
 private:

  using Real = numeric_t<Tp>;
  const std::size_t m_max_iter;
  std::size_t m_curr_iter;
  Real m_toler;

 public:

  Terminator(std::size_t max_iter, Real mul = Real{1})
      : m_max_iter(max_iter), m_curr_iter{0},
        m_toler(std::abs(mul) * std::numeric_limits<Real>::epsilon()) {}

  /// Return the current number of terms summed.
  std::size_t
  num_terms() const { return this->m_curr_iter; }

  /// Detect if the sum should terminate either because the incoming term
  /// is small enough or the maximum number of terms has been reached.
  bool
  operator()(Tp term, Tp sum) {
    if (this->m_curr_iter >= this->m_max_iter
        || ++this->m_curr_iter == this->m_max_iter)
      return true;
    else if (std::abs(term) < this->m_toler * std::abs(sum))
      return true;
    else
      return false;
  }
};

template<typename Tp>
class AsympTerminator {
 private:

  using Real = numeric_t<Tp>;
  const std::size_t m_max_iter;
  std::size_t m_curr_iter;
  Real m_toler;
  Real m_prev_term = std::numeric_limits<Real>::max();
  bool m_stop_asymp = false;

 public:

  AsympTerminator(std::size_t max_iter, Real mul = Real{1})
      : m_max_iter(max_iter), m_curr_iter{0},
        m_toler(std::abs(mul) * std::numeric_limits<Real>::epsilon()) {}

  /// Filter a term before applying it to the sum.
  Tp
  operator<<(Tp term) {
    if (std::abs(term) > this->m_prev_term) {
      this->m_stop_asymp = true;
      return Tp{0};
    } else
      return term;
  }

  /// Return the current number of terms summed.
  std::size_t
  num_terms() const { return this->m_curr_iter; }

  /// Detect if the sum should terminate either because the incoming term
  /// is small enough or because the terms are starting to grow or
  //  the maximum number of terms has been reached.
  bool
  operator()(Tp term, Tp sum) {
    if (this->m_stop_asymp)
      return true;
    else {
      const auto aterm = std::abs(term);
      this->m_stop_asymp = (aterm > this->m_prev_term);
      this->m_prev_term = aterm;
      if (this->m_curr_iter >= this->m_max_iter
          || ++this->m_curr_iter == this->m_max_iter)
        return true;
      else if (aterm < this->m_toler * std::abs(sum))
        return true;
      else if (this->m_stop_asymp)
        return true;
      else
        return false;
    }
  }
};

template<typename Tp>
std::complex<Tp>
clamp_pi(std::complex<Tp> z) {
  using Real = numeric_t<Tp>;
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_i2pi = std::complex<Tp>{0, Tp{2} * s_pi};
  while (z.imag() > s_pi)
    z -= s_i2pi;
  while (z.imag() <= -s_pi)
    z += s_i2pi;
  return z;
}

template<typename Tp>
std::complex<Tp>
clamp_0_m2pi(std::complex<Tp> z) {
  using Real = numeric_t<Tp>;
  const auto s_2pi = c10::numbers::tau_v<Real>;
  while (z.imag() > Tp{0})
    z = std::complex<Tp>(z.real(), z.imag() - s_2pi);
  while (z.imag() <= -s_2pi)
    z = std::complex<Tp>(z.real(), z.imag() + s_2pi);
  return z;
}

template<typename Tp>
std::complex<Tp>
polylog_exp_pos(unsigned int s, std::complex<Tp> w) {
  using Real = numeric_t<Tp>;
  const auto s_2pi = c10::numbers::tau_v<Real>;
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_pipio6 = c10::numbers::pi_sqr_div_6_v<Real>;
  std::complex<Tp> res = riemann_zeta<Tp>(s);
  auto wk = w;
  auto fact = Tp{1};
  auto harmonicN = Tp{1}; // HarmonicNumber_1
  for (unsigned int k = 1; k <= s - 2; ++k) {
    res += fact * riemann_zeta<Tp>(s - k) * wk;
    wk *= w;
    const auto temp = Tp{1} / Tp(1 + k);
    fact *= temp;
    harmonicN += temp;
  }
  // harmonicN now contains H_{s-1}.
  // fact should be 1/(s-1)!
  res += (harmonicN - std::log(-w)) * wk * fact;
  wk *= w;
  fact /= s; // 1/s!
  res -= wk * fact / Tp{2};
  wk *= w;
  // Now comes the remainder of the series.
  const auto pref = wk / s_pi / s_2pi;
  fact /= Tp(s + 1); // 1/(s+1)!
  // Subtract the zeroth order term.
  res -= s_pipio6 * fact * pref;
  fact *= Tp{2} / Tp(s + 2) * Tp{3} / Tp(s + 3);
  const auto wbar = w / s_2pi;
  const auto w2 = -wbar * wbar;
  auto w2k = w2;
  auto rzarg = Tp{2};
  const unsigned int maxit = 200;
  Terminator<std::complex<Tp>> done(maxit);
  while (true) {
    rzarg += Tp{2};
    const auto rzeta = riemann_zeta(rzarg);
    const auto term = pref * fact * rzeta * w2k;
    res -= term;
    if (done(term, res))
      break;
    w2k *= w2;
    fact *= Tp(rzarg) / Tp(s + rzarg)
        * Tp(rzarg + 1) / Tp(s + rzarg + 1);
  }
  return res;
}

template<typename Tp>
std::complex<Tp>
polylog_exp_pos(unsigned int s, Tp w) {
  const auto s_2pi = c10::numbers::tau_v<Tp>;
  const auto s_pi = c10::numbers::pi_v<Tp>;
  const auto s_pipio6 = c10::numbers::pi_sqr_div_6_v<Tp>;
  auto res = riemann_zeta<Tp>(s);
  auto wk = w;
  auto fact = Tp{1};
  auto harmonicN = Tp{1}; // HarmonicNumber_1
  for (unsigned int k = 1; k <= s - 2; ++k) {
    res += fact * riemann_zeta<Tp>(s - k) * wk;
    wk *= w;
    const auto temp = Tp{1} / Tp(1 + k);
    fact *= temp;
    harmonicN += temp;
  }
  // harmonicN now contains H_{s-1}
  // fact should be 1/(s-1)!
  const auto imagtemp = fact * wk
      * (harmonicN - std::log(std::complex<Tp>(-w)));
  res += std::real(imagtemp);
  wk *= w;
  fact /= s; // 1/s!
  res -= wk * fact / Tp{2};
  wk *= w;
  // Now comes the remainder of the series.
  const auto pref = wk / s_pi / s_2pi;
  fact /= Tp(s + 1); // 1/(s+1)!
  // Subtract the zeroth order term.
  res -= s_pipio6 * fact * pref;
  fact *= Tp{2} / Tp(s + 2) * Tp{3} / Tp(s + 3);
  const auto wbar = w / s_2pi;
  const auto w2 = -wbar * wbar;
  auto w2k = w2;
  auto rzarg = Tp{2};
  const unsigned int maxit = 200;
  Terminator<Tp> done(maxit);
  while (true) {
    rzarg += Tp{2};
    const auto rzeta = riemann_zeta(rzarg);
    const auto term = pref * fact * rzeta * w2k;
    res -= term;
    if (done(term, res))
      break;
    w2k *= w2;
    fact *= Tp(rzarg) / Tp(s + rzarg)
        * Tp(rzarg + 1) / Tp(s + rzarg + 1);
  }
  return std::complex<Tp>(res, std::imag(imagtemp));
}

template<typename Tp>
std::complex<Tp>
polylog_exp_neg(Tp s, std::complex<Tp> w) {
  const auto s_i = std::complex<Tp>{0, 1};
  const auto s_2pi = c10::numbers::tau_v<Tp>;
  // Basic general loop, but s is a negative quantity here
  // FIXME Large s makes problems.
  // The series should be rearrangeable so that we only need
  // the ratio Gamma(1-s)/(2 pi)^s
  auto ls = ln_gamma(Tp{1} - s);
  auto res = std::exp(ls - (Tp{1} - s) * std::log(-w));
  const auto wup = w / s_2pi;
  auto w2k = wup;
  const auto pref = Tp{2} * std::pow(s_2pi, -(Tp{1} - s));
  // Here we factor up the ratio of Gamma(1 - s + k)/k! .
  // This ratio should be well behaved even for large k in the series
  // afterwards
  // Note that we have a problem for large s.
  // Since s is negative we evaluate the Gamma Function
  // on the positive real axis where it is real.
  auto gam = std::exp(ls);

  const auto phase = polar_pi(Tp{1}, s / Tp{2});
  const auto cp = std::real(phase);
  const auto sp = std::imag(phase);
  // Here we add the expression that would result from ignoring
  // the zeta function in the series.
  const auto p = s_2pi - s_i * w;
  const auto q = s_2pi + s_i * w;
  // This can be optimized for real values of w
  res += s_i * gam * (std::conj(phase) * std::pow(p, s - Tp{1})
      - phase * std::pow(q, s - Tp{1}));
  // The above expression is the result of
  // sum_k Gamma(1+k-s)/k! * sin(pi (s-k)/2) (w/2/pi)^k
  // Therefore we only need to sample values of zeta(n) on the real axis
  // that really differ from one
  std::complex<Tp> sum = sp * gam * riemann_zeta_m_1(Tp{1} - s);
  unsigned int j = 1;
  gam *= (Tp{1} - s);
  constexpr unsigned int maxit = 200;
  Terminator<std::complex<Tp>> done(maxit);
  while (true) {
    const auto rzarg = Tp(1 + j) - s;
    const auto rz = riemann_zeta_m_1(rzarg);
    Tp sine;
    // Save repeated recalculation of the sines.
    if (j & 1) { // odd
      sine = cp;
      if (!((j - 1) / 2 & 1))
        sine = -sine;
    } else { // even
      sine = sp;
      if ((j / 2) & 1)
        sine = -sine;
    }
    const auto term = w2k * (gam * sine * rz);
    w2k *= wup;
    ++j;
    gam *= rzarg / Tp(j); // == 1/(j+1) we incremented j above.
    sum += term;
    if (done(term, sum))
      break;
  }
  res += pref * sum;
  return res;
}

template<typename Tp>
std::complex<Tp>
polylog_exp_neg(int n, std::complex<Tp> w) {
  const auto s_inf = std::numeric_limits<Tp>::infinity();
  if (is_zero(w))
    return std::complex<Tp>{0};
  else if (is_equal(w, Tp{1}))
    return std::complex<Tp>{s_inf, Tp{0}};
  else {
    const int p = -n;
    const int pp = 1 + p;
    const int q = p & 1 ? 0 : 1;
    const auto w2 = w * w;
    auto wp = p & 1 ? std::complex<Tp>{1} : w;
    unsigned int __2k = q;
    auto gam = factorial<Tp>(p + __2k);
    const auto pfact = factorial<Tp>(p);
    auto res = pfact * std::pow(-w, Tp(-pp));
    auto sum = std::complex<Tp>{};
    constexpr unsigned int maxit = 300;
    Terminator<std::complex<Tp>> done(maxit);
    while (true) {
      const auto id = (p + __2k + 1) / 2;
      if (id == EULER_MACLAURIN_SIZE)
        break;
      const auto term = gam * wp
          * Tp(EULER_MACLAURIN[id]);
      sum += term;
      if (done(term, sum))
        break;
      gam *= Tp(p + __2k + 1) / Tp(__2k + 1)
          * Tp(p + __2k + 2) / Tp(__2k + 2);
      wp *= w2;
      __2k += 2;
    }
    res -= sum;
    return res;
  }
}

template<typename Tp>
std::complex<Tp>
polylog_exp_pos(Tp s, std::complex<Tp> w) { // positive s
  const auto s_2pi = c10::numbers::tau_v<Tp>;
  const auto s_pi = c10::numbers::pi_v<Tp>;
  std::complex<Tp> res = riemann_zeta(s);
  auto wk = w;
  const auto phase = polar_pi(Tp{1}, s / Tp{2});
  const auto cp = std::real(phase);
  const auto sp = std::imag(phase);
  // This is \Gamma(1-s)(-w)^{s-1}
  res += s_pi / (Tp{2} * sp * cp)
      * std::exp(-ln_gamma(s) + (s - Tp{1}) * std::log(-w));
  auto fact = Tp{1};
  const auto m = static_cast<unsigned int>(std::floor(s));
  for (unsigned int k = 1; k <= m; ++k) {
    res += wk * fact
        * riemann_zeta(s - Tp(k));
    wk *= w;
    fact /= Tp(1 + k);
  }
  // fac should now be 1/(m+1)!
  const auto pref = Tp{2} * std::pow(s_2pi, s - Tp{1});
  // Factor this out for now so we can compare with sum.
  res /= pref;
  // Now comes the remainder of the series
  unsigned int j = 0;
  constexpr unsigned int maxit = 100;
  Terminator<std::complex<Tp>> done(maxit);
  auto wup = w / s_2pi;
  auto wbark = std::pow(wup, Tp(m + 1));
  // It is 1 < 2 - s + m < 2 => Gamma(2-s+m) will not overflow
  // Here we factor up the ratio of Gamma(1 - s + k) / k!.
  // This ratio should be well behaved even for large k
  auto gam = gamma(Tp(2 + m) - s) * fact;
  std::complex<Tp> sum{};
  while (true) {
    const auto idx = m + 1 + j;
    const auto zetaarg = Tp(1 + idx) - s;
    const auto rz = riemann_zeta(zetaarg);
    auto sine = cp;
    if (idx & 1) // Save the repeated calculation of the sines.
    { // odd
      sine = cp;
      if (!((idx - 1) / 2 & 1))
        sine = -sine;
    } else { // even
      sine = sp;
      if ((idx / 2) & 1)
        sine = -sine;
    }
    const auto term = wbark * sine * gam * rz;
    wbark *= wup;
    gam *= zetaarg / Tp(1 + idx);
    ++j;
    sum += term;
    if (done(term, res + sum))
      break;
  }
  res += sum;
  return pref * res;
}

template<typename Tp>
std::complex<Tp>
polylog_exp_asymp(Tp s, std::complex<Tp> w) {
  const auto s_pi = c10::numbers::pi_v<Tp>;
  // wgamma = w^{s-1} / Gamma(s)
  auto wgamma = std::pow(w, s - Tp{1}) * reciprocal_gamma(s);
  auto res = std::complex<Tp>(Tp{0}, -s_pi) * wgamma;
  // wgamma = w^s / Gamma(s+1)
  wgamma *= w / s;
  constexpr unsigned int maxiter = 100;
  AsympTerminator<std::complex<Tp>> done(maxiter);
  // zeta(0) w^s / Gamma(s + 1)
  std::complex<Tp> oldterm = -Tp{0.5L} * wgamma;
  res += Tp{2} * oldterm;
  std::complex<Tp> term;
  auto wq = Tp{1} / (w * w);
  int k = 1;
  while (true) {
    wgamma *= wq * (s + Tp(1 - 2 * k)) * (s + Tp(2 - 2 * k));
    term = riemann_zeta<Tp>(2 * k) * wgamma;
    res += done << Tp{2} * term;
    if (done(Tp{2} * term, res))
      break;
    oldterm = term;
    ++k;
  }
  return res;
}

template<typename PowTp, typename Tp>
Tp
polylog_exp_sum(PowTp s, Tp w) {
  auto ew = std::exp(w);
  const auto up = ew;
  auto res = ew;
  unsigned int maxiter = 500;
  Terminator<Tp> done(maxiter);
  bool terminate = false;
  unsigned int k = 2;
  while (!terminate) {
    ew *= up;
    Tp temp = std::pow(k, s); // This saves us a type conversion
    const auto term = ew / temp;
    res += term;
    terminate = done(term, res);
    ++k;
  }
  return res;
}

template<typename Tp>
std::complex<Tp>
polylog_exp_pos_int(unsigned int s, std::complex<Tp> w) {
  using Val = Tp;
  using Real = numeric_t<Val>;
  const auto s_2pi = c10::numbers::tau_v<Real>;
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_pi_2 = c10::numbers::pi_v<Real> / Real{2};
  const auto s_max_asymp = Tp{5};
  const auto rw = w.real();
  const auto iw = w.imag();
  if (is_real(w)
      && is_equal(std::remainder(iw, s_2pi), Tp{0})) {
    if (s == 1)
      return std::numeric_limits<Tp>::infinity();
    else
      return riemann_zeta<Tp>(s);
  } else if (0 == s) {
    const auto t = std::exp(w);
    return is_zero(Tp{1} - t)
           ? std::numeric_limits<Tp>::quiet_NaN()
           : t / (Tp{1} - t);
  } else if (1 == s) {
    const auto t = std::exp(w);
    return is_zero(Tp{1} - t)
           ? std::numeric_limits<Tp>::quiet_NaN()
           : -std::log(Tp{1} - t);
  } else {
    if (rw < -(s_pi_2 + s_pi / Tp{5}))
      // Choose the exponentially converging series
      return polylog_exp_sum(s, w);
    else if (rw < s_max_asymp)
      // The transition point chosen here, is quite arbitrary
      // and needs more testing.
      // The reductions of the imaginary part yield the same results
      // as Mathematica.
      // Necessary to improve the speed of convergence
      return polylog_exp_pos(s, clamp_pi(w));
    else
      // Wikipedia says that this is required for Wood's formula.
      return polylog_exp_asymp(static_cast<Tp>(s),
                               clamp_0_m2pi(w));
  }
}

template<typename T1>
std::complex<T1>
polylog_exp_pos_int(unsigned int s, T1 w) {
  using Val = T1;
  using Real = numeric_t<Val>;
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_pi_2 = c10::numbers::pi_v<Real> / Real{2};
  const auto s_max_asymp = T1{5};
  if (is_zero(w)) {
    if (s == 1)
      return std::numeric_limits<T1>::infinity();
    else
      return riemann_zeta<T1>(s);
  } else if (s == 0) {
    const auto t = std::exp(w);
    return is_zero(T1{1} - t)
           ? std::numeric_limits<T1>::infinity()
           : t / (T1{1} - t);
  } else if (s == 1) {
    const auto t = std::exp(w);
    return is_zero(T1{1} - t)
           ? -std::numeric_limits<T1>::infinity()
           : -std::log(T1{1} - t);
  } else {
    if (w < -(s_pi_2 + s_pi / T1{5}))
      // Choose the exponentially converging series
      return polylog_exp_sum(s, w);
    else if (w < s_max_asymp)
      return polylog_exp_pos(s, w);
    else
      return polylog_exp_asymp(static_cast<T1>(s),
                               std::complex<T1>(w));
  }
}

template<typename Tp>
std::complex<Tp>
polylog_exp_neg_int(int s, std::complex<Tp> w) {
  using Val = Tp;
  using Real = numeric_t<Val>;
  const auto s_2pi = c10::numbers::tau_v<Real>;
  const auto s_pi = c10::numbers::pi_v<Real>;
  const auto s_pi_2 = c10::numbers::pi_v<Real> / Real{2};
  const auto s_max_asymp = Tp{5};
  if ((((-s) & 1) == 0) && is_imag(w)) {
    // Now s is odd and w on the unit-circle.
    const auto iw = imag(w); // Get imaginary part.
    const auto rem = std::remainder(iw, s_2pi);
    if (is_equal(std::abs(rem), Tp{0.5L}))
      // Due to: Li_{-n}(-1) + (-1)^n Li_{-n}(1/-1) = 0.
      return Tp{0};
    else
      // No asymptotic expansion available... check the reduction.
      return polylog_exp_neg(s, std::complex<Tp>(w.real(), rem));
  } else {
    if (std::real(w) < -(s_pi_2 + s_pi / Tp{5}))
      // Choose the exponentially converging series
      return polylog_exp_sum(s, w);
    else if (std::real(w) < s_max_asymp)
      // Arbitrary transition point...
      // The reductions of the imaginary part yield the same results
      // as Mathematica.
      // Necessary to improve the speed of convergence.
      return polylog_exp_neg(s, clamp_pi(w));
    else
      // Wikipedia says that this clamping is required for Wood's formula.
      return polylog_exp_asymp(Tp(s), clamp_0_m2pi(w));
  }
}

template<typename Tp>
std::complex<Tp>
polylog_exp_neg_int(int s, Tp w) {
  const auto s_pi = c10::numbers::pi_v<Tp>;
  const auto s_pi_2 = c10::numbers::pi_v<Tp> / Tp{2};
  const auto s_max_asymp = Tp{5};
  if (w < -(s_pi_2 + s_pi / Tp{5}))
    // Choose exponentially converging series.
    return polylog_exp_sum(s, w);
  else if (is_zero(w))
    return std::numeric_limits<Tp>::infinity();
  else if (w < s_max_asymp)
    // Arbitrary transition point less than 2 pi.
    return polylog_exp_neg(s, std::complex<Tp>(w));
  else
    return polylog_exp_asymp(Tp(s), std::complex<Tp>(w));
}

template<typename T1>
std::complex<T1>
polylog_exp_pos_real(T1 s, std::complex<T1> w) {
  if (is_real(w)
      && is_zero(std::remainder(w.imag(), c10::numbers::tau_v<T1>))) {
    if (is_equal(s, T1(1)))
      return std::numeric_limits<T1>::infinity();
    else
      return riemann_zeta(s);
  }

  if (w.real() < -(c10::numbers::pi_v<T1> / T1(2) + c10::numbers::pi_v<T1> / T1(5)))
    return polylog_exp_sum(s, w);

  if (w.real() < T1(5))
    return polylog_exp_pos(s, clamp_pi(w));
  else
    return polylog_exp_asymp(s, clamp_0_m2pi(w));
}

template<typename T1>
std::complex<T1>
polylog_exp_pos_real(T1 s, T1 w) {
  if (is_zero(w)) {
    if (is_equal(s, T1(1))) {
      return std::numeric_limits<T1>::infinity();
    } else {
      return riemann_zeta(s);
    }
  } else if (w < -(c10::numbers::pi_v<T1> / T1(2) + c10::numbers::pi_v<T1> / T1(5))) {
    return polylog_exp_sum(s, w);
  } else if (w < T1(5)) {
    return polylog_exp_pos(s, std::complex<T1>(w));
  } else {
    return polylog_exp_asymp(s, std::complex<T1>(w));
  }
}

template<typename Tp>
std::complex<Tp>
polylog_exp_neg_real(Tp s, std::complex<Tp> w) {
  const auto s_pi = c10::numbers::pi_v<Tp>;
  const auto s_pi_2 = c10::numbers::pi_v<Tp> / Tp{2};
  const auto s_max_asymp = Tp{5};
  const auto rw = w.real();
  if (rw < -(s_pi_2 + s_pi / Tp{5}))
    return polylog_exp_sum(s, w);
  else if (rw < s_max_asymp)
    return polylog_exp_neg(s, clamp_pi(w));
  else
    return polylog_exp_asymp(s, clamp_0_m2pi(w));
}

template<typename T1>
std::complex<T1>
polylog_exp_neg_real(T1 s, T1 w) {
  if (w < -(c10::numbers::pi_v<T1> / T1(2) + c10::numbers::pi_v<T1> / T1(5))) {
    return polylog_exp_sum(s, w);
  } else if (w < T1(5)) {
    return polylog_exp_neg(s, std::complex<T1>(w));
  } else {
    return polylog_exp_asymp(s, std::complex<T1>(w));
  }
}

template<typename T1, typename T2>
promote_t<std::complex<T1>, T2>
exp_polylog(T1 s, T2 w) {
  const auto is_integer_s = is_integer(s, T1(5));

  if (std::isnan(s) || std::isnan(w)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else if (s > T1(25)) {
    return polylog_exp_sum(s, w);
  } else if (is_integer_s) {
    if (is_integer_s() >= 0) {
      return polylog_exp_pos_int(is_integer_s(), w);
    } else {
      return polylog_exp_neg_int(is_integer_s(), w);
    }
  } else {
    if (s > T1(0)) {
      return polylog_exp_pos_real(s, w);
    } else {
      return polylog_exp_neg_real(s, w);
    }
  }
}
}
}
}
}
