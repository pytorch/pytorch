#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace fft {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor fft(const Tensor& self) {
  return torch::fft_fft(self);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation of torch.fft.fft.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::fft(t);
/// ```
inline Tensor fft(const Tensor& self) {
  return detail::fft(self);
}

}} // torch::fft
