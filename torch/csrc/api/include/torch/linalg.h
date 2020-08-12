#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace linalg {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor outer(const Tensor& self, const Tensor& vec2) {
  return torch::linalg_outer(self, vec2);
}

inline Tensor outer_out(Tensor &result, const Tensor& self, const Tensor& vec2) {
  return torch::linalg_outer_out(result, self, vec2);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See the documentation of torch.linalg.outer
inline Tensor outer(const Tensor& self, const Tensor& vec2) {
  return detail::outer(self, vec2);
}

/// See the documentation of torch.linalg.outer
inline Tensor outer_out(Tensor &result, const Tensor& self, const Tensor& vec2) {
  return detail::outer_out(result, self, vec2);
}

}} // torch::linalg
