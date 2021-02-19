#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace special {

/// Computes the logarithm of the gamma function on :attr:`input`.
/// See https://pytorch.org/docs/master/special.html#torch.special.gammaln.
///
/// Example:
/// ```
/// auto t = torch::randn(128, dtype=kDouble);
/// torch::special::gammaln(t);
/// ```
inline Tensor gammaln(const Tensor& self) {
  return torch::special_gammaln(self);
}

}} // torch::special
