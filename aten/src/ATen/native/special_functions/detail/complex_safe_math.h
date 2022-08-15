#pragma once

#include <complex>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename Tp>
std::complex<Tp>
safe_div(const std::complex<Tp> &z1, const std::complex<Tp> &z2);

template<typename _Sp, typename Tp>
inline constexpr std::complex<Tp>
safe_div(_Sp s, const std::complex<Tp> &z) { return safe_div(std::complex<Tp>(s), z); }

template<typename _Sp, typename Tp>
inline constexpr std::complex<Tp>
safe_div(const std::complex<Tp> &z, _Sp s) { return safe_div(z, std::complex<Tp>(s)); }

template<typename Tp>
Tp
safe_mul(Tp s1, Tp s2);

template<typename Tp>
std::complex<Tp>
safe_mul(const std::complex<Tp> &z1, const std::complex<Tp> &z2);

template<typename _Sp, typename Tp>
inline constexpr std::complex<Tp>
safe_mul(_Sp s, const std::complex<Tp> &z) { return safe_mul(std::complex<Tp>(s), z); }

template<typename _Sp, typename Tp>
inline constexpr std::complex<Tp>
safe_mul(const std::complex<Tp> &z, _Sp s) { return safe_mul(z, std::complex<Tp>(s)); }

template<typename Tp>
std::complex<Tp>
safe_sqr(const std::complex<Tp> &z);

template<typename Tp>
std::complex<Tp>
safe_div(const std::complex<Tp> &z1, const std::complex<Tp> &z2) {
  // Half the largest available floating-point number.
  const auto s_hmax = emsr::lim_max(std::real(z1)) / Tp{2};

  auto re1 = std::real(z1);
  auto im1 = std::imag(z1);
  auto re2 = std::real(z2);
  auto im2 = std::imag(z2);

  //  Find the largest and smallest magnitudes
  auto z1b = std::max(std::abs(re1), std::abs(im1));
  auto z2max = std::abs(re2);
  auto z2min = std::abs(im2);
  if (z2max < z2min)
    std::swap(z2max, z2min);

  if (z2max < Tp{1} && z1b > z2max * s_hmax)
    throw std::runtime_error(__N("safe_div: overflow in complex division"));

  re1 /= z1b;
  im1 /= z1b;
  re2 /= z2max;
  im2 /= z2max;
  auto term = z2min / z2max;
  auto denom = Tp{1} + term * term;
  auto scale = z1b / z2max / denom;
  auto qr = (re1 * re2 + im1 * im2) * scale;
  auto qi = (re2 * im1 - re1 * im2) * scale;

  return std::complex<Tp>{qr, qi};
}

template<typename Tp>
Tp
safe_mul(Tp s1, Tp s2) {
  // The largest available floating-point number.
  const auto s_max = emsr::lim_max(std::real(s1));
  const auto s_sqrt_max = emsr::sqrt_max(std::real(s1));
  auto abs_s1 = std::abs(s1);
  auto abs_s2 = std::abs(s2);
  if (abs_s1 < s_sqrt_max || abs_s2 < s_sqrt_max) {
    auto abs_max = abs_s1;
    auto abs_min = abs_s2;
    if (abs_max < abs_min)
      std::swap(abs_max, abs_min);
    if (abs_max > s_sqrt_max && abs_min > s_max / abs_max)
      throw std::runtime_error(__N("safe_mul: overflow in scalar multiplication"));
    else
      return s1 * s2;
  } else
    throw std::runtime_error(__N("safe_mul: overflow in scalar multiplication"));
}

template<typename Tp>
std::complex<Tp>
safe_mul(const std::complex<Tp> &z1, const std::complex<Tp> &z2) {
  // Half the largest available floating-point number.
  const auto s_max = emsr::lim_max(std::real(z1));
  const auto s_sqrt_max = emsr::sqrt_max(std::real(z1));

  auto re1 = std::real(z1);
  auto im1 = std::imag(z1);
  auto re2 = std::real(z2);
  auto im2 = std::imag(z2);

  auto abs_rem = std::abs(re1 - im1);
  auto abs_rep = std::abs(re2 + im2);
  if (abs_rem < s_sqrt_max || abs_rep < s_sqrt_max) {
    // Find the largest and smallest magnitudes
    auto abs_min = abs_rem;
    auto abs_max = abs_rep;
    if (abs_max < abs_min)
      std::swap(abs_max, abs_min);
    if (abs_max > s_sqrt_max && abs_min > s_max / abs_max)
      throw std::runtime_error(__N("safe_mul: overflow in complex multiplication"));
    else
      return std::complex<Tp>((re1 - im1) * (re2 + im2),
                              safe_mul(re1, im2) + safe_mul(re2, im1));
  } else
    throw std::runtime_error(__N("safe_mul: overflow in complex multiplication"));
}

template<typename Tp>
std::complex<Tp>
safe_sqr(const std::complex<Tp> &z) {
  const auto s_sqrt_2 = emsr::sqrt2_v<Tp>;
  const auto s_max = emsr::lim_max<Tp>();
  const auto s_hmax = s_max / Tp{2};
  const auto s_sqrt_max = emsr::sqrt_max<Tp>();
  const auto s_sqrt_hmax = s_sqrt_max / s_sqrt_2;

  auto rez = std::real(z);
  auto imz = std::imag(z);
  auto abs_rez = std::abs(rez);
  auto abs_imz = std::abs(imz);
  auto zm = rez - imz;
  auto zp = rez + imz;
  auto abs_zm = std::abs(zm);
  auto abs_zp = std::abs(zp);

  if ((abs_zm < s_sqrt_max || abs_zp < s_sqrt_max)
      && (abs_rez < s_sqrt_hmax || abs_imz < s_sqrt_hmax)) {
    // Sort the magnitudes of the imag part factors.
    auto imzmax = abs_rez;
    auto imzmin = abs_imz;
    if (imzmax < imzmin)
      std::swap(imzmax, imzmin);
    if (imzmax >= s_sqrt_hmax && imzmin > s_hmax / imzmax)
      throw std::runtime_error(__N("safe_sqr: overflow in complex multiplication"));

    // Sort the magnitudes of the real part factors.
    auto rezmax = abs_zp;
    auto rezmin = abs_zm;
    if (imzmax < rezmin)
      std::swap(rezmax, rezmin);
    if (rezmax >= s_sqrt_max && rezmin > s_max / rezmax)
      throw std::runtime_error(__N("safe_sqr: "
                                   "overflow in complex multiplication"));

    return std::complex<Tp>(zm * zp, Tp{2} * rez * imz);
  } else
    throw std::runtime_error(__N("safe_sqr: overflow in complex multiplication"));
}
}
}
}
}
