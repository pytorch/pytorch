#pragma once

#include <complex>

namespace at::native::special_functions::detail {
template<typename Tp>
inline constexpr Tp
l1_norm(const std::complex<Tp> &z) { return std::abs(std::real(z)) + std::abs(std::imag(z)); }

template<typename Tp>
inline constexpr Tp
l2_norm(const std::complex<Tp> &z) { return std::norm(z); }

template<typename Tp>
inline constexpr Tp
linf_norm(const std::complex<Tp> &z) { return std::max(std::abs(std::real(z)), std::abs(std::imag(z))); }

template<typename Tp>
inline constexpr Tp
l1_norm(Tp x) { return std::abs(x); }

template<typename Tp>
inline constexpr Tp
l2_norm(Tp x) { return std::abs(x); }

template<typename Tp>
inline constexpr Tp
linf_norm(Tp x) { return std::abs(x); }

}
