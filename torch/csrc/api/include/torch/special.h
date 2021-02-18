#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace special {

inline Tensor gammaln(const Tensor& self) {
  return torch::special_gammaln(self);
}

}} // torch::special
