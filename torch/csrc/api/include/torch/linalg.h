#pragma once

#include <ATen/ATen.h>

namespace torch {
namespace linalg {

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {

inline Tensor cholesky(const Tensor& self) {
  return torch::linalg_cholesky(self);
}

inline Tensor cholesky_out(Tensor& result, const Tensor& self) {
  return torch::linalg_cholesky_out(result, self);
}

inline Tensor det(const Tensor& self) {
  return torch::linalg_det(self);
}

inline std::tuple<Tensor, Tensor> slogdet(const Tensor& input) {
  return torch::linalg_slogdet(input);
}

inline std::tuple<Tensor&, Tensor&> slogdet_out(Tensor& sign, Tensor& logabsdet, const Tensor& input) {
  return torch::linalg_slogdet_out(sign, logabsdet, input);
}

inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return torch::linalg_eig(self);
}

inline std::tuple<Tensor&, Tensor&> eig_out(Tensor& eigvals, Tensor& eigvecs, const Tensor& self) {
  return torch::linalg_eig_out(eigvals, eigvecs, self);
}

inline Tensor eigvals(const Tensor& self) {
  return torch::linalg_eigvals(self);
}

inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return torch::linalg_eigvals_out(result, self);
}

inline std::tuple<Tensor, Tensor> eigh(const Tensor& self, std::string uplo) {
  return torch::linalg_eigh(self, uplo);
}

inline std::tuple<Tensor&, Tensor&> eigh_out(Tensor& eigvals, Tensor& eigvecs, const Tensor& self, std::string uplo) {
  return torch::linalg_eigh_out(eigvals, eigvecs, self, uplo);
}

inline Tensor eigvalsh(const Tensor& self, std::string uplo) {
  return torch::linalg_eigvalsh(self, uplo);
}

inline Tensor& eigvalsh_out(Tensor& result, const Tensor& self, std::string uplo) {
  return torch::linalg_eigvalsh_out(result, self, uplo);
}

inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return torch::linalg_householder_product(input, tau);
}

inline Tensor& householder_product_out(Tensor& result, const Tensor& input, const Tensor& tau) {
  return torch::linalg_householder_product_out(result, input, tau);
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& self, const Tensor& b, c10::optional<double> cond, c10::optional<std::string> driver) {
  return torch::linalg_lstsq(self, b, cond, driver);
}

inline Tensor norm(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor vector_norm(const Tensor& self, Scalar ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& vector_norm_out(Tensor& result, const Tensor& self, Scalar ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor matrix_norm(const Tensor& self, const Scalar& ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(const Tensor& self, const Scalar& ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype, Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

inline Tensor matrix_norm(const Tensor& self, std::string ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(const Tensor& self, std::string ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype, Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return torch::linalg_matrix_power(self, n);
}

inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return torch::linalg_matrix_power_out(result, self, n);
}

inline Tensor matrix_rank(const Tensor input, optional<double> tol, bool hermitian) {
  return torch::linalg_matrix_rank(input, tol, hermitian);
}

inline Tensor& matrix_rank_out(Tensor& result, const Tensor input, optional<double> tol, bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, tol, hermitian);
}

inline Tensor multi_dot(TensorList tensors) {
  return torch::linalg_multi_dot(tensors);
}

inline Tensor& multi_dot_out(TensorList tensors, Tensor& result) {
  return torch::linalg_multi_dot_out(result, tensors);
}

inline Tensor pinv(const Tensor& input, double rcond, bool hermitian) {
  return torch::linalg_pinv(input, rcond, hermitian);
}

inline Tensor& pinv_out(Tensor& result, const Tensor& input, double rcond, bool hermitian) {
  return torch::linalg_pinv_out(result, input, rcond, hermitian);
}

inline Tensor solve(const Tensor& input, const Tensor& other) {
  return torch::linalg_solve(input, other);
}

inline Tensor& solve_out(Tensor& result, const Tensor& input, const Tensor& other) {
  return torch::linalg_solve_out(result, input, other);
}

inline std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& input, bool full_matrices) {
  return torch::linalg_svd(input, full_matrices);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(Tensor& U, Tensor& S, Tensor& Vh, const Tensor& input, bool full_matrices) {
  return torch::linalg_svd_out(U, S, Vh, input, full_matrices);
}

inline Tensor svdvals(const Tensor& input) {
  return torch::linalg_svdvals(input);
}

inline Tensor& svdvals_out(Tensor& result, const Tensor& input) {
  return torch::linalg_svdvals_out(result, input);
}

inline Tensor tensorinv(const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv(self, ind);
}

inline Tensor& tensorinv_out(Tensor& result,const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv_out(result, self, ind);
}

inline Tensor tensorsolve(const Tensor& self, const Tensor& other, optional<IntArrayRef> dims) {
  return torch::linalg_tensorsolve(self, other, dims);
}

inline Tensor& tensorsolve_out(Tensor& result, const Tensor& self, const Tensor& other, optional<IntArrayRef> dims) {
  return torch::linalg_tensorsolve_out(result, self, other, dims);
}

inline Tensor inv(const Tensor& input) {
  return torch::linalg_inv(input);
}

inline Tensor& inv_out(Tensor& result, const Tensor& input) {
  return torch::linalg_inv_out(result, input);
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

inline Tensor cholesky_out(Tensor& result, const Tensor& self) {
  return detail::cholesky_out(result, self);
}

/// DEPRECATED
inline Tensor linalg_det(const Tensor& self) {
  return detail::det(self);
}

/// See the documentation of torch.linalg.det
inline Tensor det(const Tensor& self) {
  return detail::det(self);
}

/// Computes the sign and (natural) logarithm of the determinant
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.slogdet
inline std::tuple<Tensor, Tensor> slogdet(const Tensor& input) {
  return detail::slogdet(input);
}

inline std::tuple<Tensor&, Tensor&> slogdet_out(Tensor& sign, Tensor& logabsdet, const Tensor& input) {
  return detail::slogdet_out(sign, logabsdet, input);
}

/// Computes eigenvalues and eigenvectors of non-symmetric/non-hermitian matrices
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.eig
inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return detail::eig(self);
}

inline std::tuple<Tensor&, Tensor&> eig_out(Tensor& eigvals, Tensor& eigvecs, const Tensor& self) {
  return detail::eig_out(eigvals, eigvecs, self);
}

/// Computes eigenvalues of non-symmetric/non-hermitian matrices
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.eigvals
inline Tensor eigvals(const Tensor& self) {
  return detail::eigvals(self);
}

inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return detail::eigvals_out(result, self);
}

/// Computes eigenvalues and eigenvectors
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.eigh
inline std::tuple<Tensor, Tensor> eigh(const Tensor& self, std::string uplo) {
  return detail::eigh(self, uplo);
}

inline std::tuple<Tensor&, Tensor&> eigh_out(Tensor& eigvals, Tensor& eigvecs, const Tensor& self, std::string uplo) {
  return detail::eigh_out(eigvals, eigvecs, self, uplo);
}

/// Computes eigenvalues
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.eigvalsh
inline Tensor eigvalsh(const Tensor& self, std::string uplo) {
  return detail::eigvalsh(self, uplo);
}

inline Tensor& eigvalsh_out(Tensor& result, const Tensor& self, std::string uplo) {
  return detail::eigvalsh_out(result, self, uplo);
}

/// Computes the product of Householder matrices
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.householder_product
inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return detail::householder_product(input, tau);
}

inline Tensor& householder_product_out(Tensor& result, const Tensor& input, const Tensor& tau) {
  return detail::householder_product_out(result, input, tau);
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& self, const Tensor& b, c10::optional<double> cond, c10::optional<std::string> driver) {
  return detail::lstsq(self, b, cond, driver);
}

/// DEPRECATED
inline Tensor linalg_norm(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

/// DEPRECATED
inline Tensor linalg_norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);
}

/// DEPRECATED
inline Tensor& linalg_norm_out(Tensor& result, const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

/// DEPRECATED
inline Tensor& linalg_norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, const optional<Scalar>& opt_ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(Tensor& result, const Tensor& self, std::string ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.vector_norm
inline Tensor vector_norm(const Tensor& self, Scalar ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& vector_norm_out(Tensor& result, const Tensor& self, Scalar ord, optional<IntArrayRef> opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  return detail::vector_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.matrix_norm
inline Tensor matrix_norm(const Tensor& self, const Scalar& ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(const Tensor& self, const Scalar& ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype, Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

inline Tensor matrix_norm(const Tensor& self, std::string ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(const Tensor& self, std::string ord, IntArrayRef dim, bool keepdim, optional<ScalarType> dtype, Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.matrix_power
inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return detail::matrix_power(self, n);
}

inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return detail::matrix_power_out(self, n, result);
}

/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.matrix_rank
inline Tensor matrix_rank(const Tensor input, optional<double> tol, bool hermitian) {
  return detail::matrix_rank(input, tol, hermitian);
}

inline Tensor& matrix_rank_out(Tensor& result, const Tensor input, optional<double> tol, bool hermitian) {
  return detail::matrix_rank_out(result, input, tol, hermitian);
}

/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.multi_dot
inline Tensor multi_dot(TensorList tensors) {
  return detail::multi_dot(tensors);
}

inline Tensor& multi_dot_out(TensorList tensors, Tensor& result) {
  return detail::multi_dot_out(tensors, result);
}

/// Computes pseudo-inverse
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.pinv
inline Tensor pinv(const Tensor& input, double rcond=1e-15, bool hermitian=false) {
  return detail::pinv(input, rcond, hermitian);
}

inline Tensor& pinv_out(Tensor& result, const Tensor& input, double rcond=1e-15, bool hermitian=false) {
  return detail::pinv_out(result, input, rcond, hermitian);
}

/// Computes a tensor `x` such that `matmul(input, x) = other`.
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.solve
inline Tensor solve(const Tensor& input, const Tensor& other) {
  return detail::solve(input, other);
}

inline Tensor& solve_out(Tensor& result, const Tensor& input, const Tensor& other) {
  return detail::solve_out(result, input, other);
}

/// Computes the singular values and singular vectors
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.svd
inline std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& input, bool full_matrices) {
  return detail::svd(input, full_matrices);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(Tensor& U, Tensor& S, Tensor& Vh, const Tensor& input, bool full_matrices) {
  return detail::svd_out(U, S, Vh, input, full_matrices);
}

/// Computes the singular values
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.svdvals
inline Tensor svdvals(const Tensor& input) {
  return detail::svdvals(input);
}

inline Tensor& svdvals_out(Tensor& result, const Tensor& input) {
  return detail::svdvals_out(result, input);
}

/// Computes the inverse of a tensor
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.tensorinv
///
/// Example:
/// ```
/// auto a = torch::eye(4*6).reshape({4, 6, 8, 3});
/// int64_t ind = 2;
/// auto ainv = torch::linalg::tensorinv(a, ind);
/// ```
inline Tensor tensorinv(const Tensor& self, int64_t ind) {
  return detail::tensorinv(self, ind);
}

inline Tensor& tensorinv_out(Tensor& result, const Tensor& self, int64_t ind) {
  return detail::tensorinv_out(result, self, ind);
}

/// Computes a tensor `x` such that `tensordot(input, x, dims=x.dim()) = other`.
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.tensorsolve
///
/// Example:
/// ```
/// auto a = torch::eye(2*3*4).reshape({2*3, 4, 2, 3, 4});
/// auto b = torch::randn(2*3, 4);
/// auto x = torch::linalg::tensorsolve(a, b);
/// ```
inline Tensor tensorsolve(const Tensor& input, const Tensor& other, optional<IntArrayRef> dims) {
  return detail::tensorsolve(input, other, dims);
}

inline Tensor& tensorsolve_out(Tensor& result, const Tensor& input, const Tensor& other, optional<IntArrayRef> dims) {
  return detail::tensorsolve_out(result, input, other, dims);
}

/// Computes a tensor `inverse_input` such that `dot(input, inverse_input) = eye(input.size(0))`.
///
/// See https://pytorch.org/docs/master/linalg.html#torch.linalg.inv
inline Tensor inv(const Tensor& input) {
  return detail::inv(input);
}

inline Tensor& inv_out(Tensor& result, const Tensor& input) {
  return detail::inv_out(result, input);
}

}} // torch::linalg
