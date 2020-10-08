#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace linalg {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor cholesky(const Tensor& self) {
  return torch::linalg_cholesky(self);
}

inline Tensor det(const Tensor& self) {
  return torch::linalg_det(self);
}

inline Tensor norm(const Tensor& self, optional<Scalar> opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, optional<Scalar> opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// Cholesky decomposition
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.cholesky
///
/// Example:
/// ```
/// auto A = torch::randn({4, 4});
/// auto A = torch::matmul(A, A.t());
/// auto L = torch::linalg::cholesky(A);
/// assert(torch::allclose(torch::matmul(L, L.t()), A));
/// ```
inline Tensor cholesky(const Tensor& self) {
  return detail::cholesky(self);
}

/// See the documentation of torch.linalg.det
inline Tensor linalg_det(const Tensor& self) {
  return detail::det(self);
}

inline Tensor linalg_norm(const Tensor& self, optional<Scalar> opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor linalg_norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& linalg_norm_out(Tensor& result, const Tensor& self, optional<Scalar> opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& linalg_norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

}} // torch::linalg
