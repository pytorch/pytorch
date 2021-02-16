#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace special {

/// Computes the 1 dimensional fast Fourier transform over a given dimension.
/// See https://pytorch.org/docs/master/fft.html#torch.fft.fft.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kComplexDouble);
/// torch::fft::fft(t);
/// ```
inline Tensor lgamma(const Tensor& self) {
  return torch::special_lgamma(self);
}

}} // torch::fft