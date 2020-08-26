#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace linalg {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor det(const Tensor& self) {
  return torch::linalg_det(self);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


/// See the documentation of torch.linalg.det
inline Tensor linalg_det(const Tensor& self) {
  return detail::det(self);
}

}} // torch::linalg
