#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace special {

/// Computes the natural logarithm of the absolute value of the gamma function
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
