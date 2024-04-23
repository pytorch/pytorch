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

inline std::tuple<Tensor&, Tensor&> slogdet_out(
    Tensor& sign,
    Tensor& logabsdet,
    const Tensor& input) {
  return torch::linalg_slogdet_out(sign, logabsdet, input);
}

inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return torch::linalg_eig(self);
}

inline std::tuple<Tensor&, Tensor&> eig_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self) {
  return torch::linalg_eig_out(eigvals, eigvecs, self);
}

inline Tensor eigvals(const Tensor& self) {
  return torch::linalg_eigvals(self);
}

inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return torch::linalg_eigvals_out(result, self);
}

inline std::tuple<Tensor, Tensor> eigh(
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigh(self, uplo);
}

inline std::tuple<Tensor&, Tensor&> eigh_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigh_out(eigvals, eigvecs, self, uplo);
}

inline Tensor eigvalsh(const Tensor& self, c10::string_view uplo) {
  return torch::linalg_eigvalsh(self, uplo);
}

inline Tensor& eigvalsh_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view uplo) {
  return torch::linalg_eigvalsh_out(result, self, uplo);
}

inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return torch::linalg_householder_product(input, tau);
}

inline Tensor& householder_product_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tau) {
  return torch::linalg_householder_product_out(result, input, tau);
}

inline std::tuple<Tensor, Tensor> lu_factor(
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_factor(self, pivot);
}

inline std::tuple<Tensor&, Tensor&> lu_factor_out(
    Tensor& LU,
    Tensor& pivots,
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_factor_out(LU, pivots, self, pivot);
}

inline std::tuple<Tensor, Tensor, Tensor> lu(
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu(self, pivot);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> lu_out(
    Tensor& P,
    Tensor& L,
    Tensor& U,
    const Tensor& self,
    const bool pivot) {
  return torch::linalg_lu_out(P, L, U, self, pivot);
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(
    const Tensor& self,
    const Tensor& b,
    c10::optional<double> cond,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_lstsq(self, b, cond, driver);
}

inline Tensor matrix_exp(const Tensor& self) {
  return torch::linalg_matrix_exp(self);
}

inline Tensor norm(
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(
      result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor vector_norm(
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& vector_norm_out(
    Tensor& result,
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return torch::linalg_vector_norm_out(
      result, self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor matrix_norm(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

inline Tensor matrix_norm(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return torch::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return torch::linalg_matrix_norm_out(result, self, ord, dim, keepdim, dtype);
}

inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return torch::linalg_matrix_power(self, n);
}

inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return torch::linalg_matrix_power_out(result, self, n);
}

inline Tensor matrix_rank(const Tensor& input, double tol, bool hermitian) {
  return torch::linalg_matrix_rank(input, tol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, tol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    c10::optional<double> atol,
    c10::optional<double> rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, atol, rtol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    const c10::optional<Tensor>& atol,
    const c10::optional<Tensor>& rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank(input, atol, rtol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    double tol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, tol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, tol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    c10::optional<double> atol,
    c10::optional<double> rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, atol, rtol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const c10::optional<Tensor>& atol,
    const c10::optional<Tensor>& rtol,
    bool hermitian) {
  return torch::linalg_matrix_rank_out(result, input, atol, rtol, hermitian);
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

inline Tensor& pinv_out(
    Tensor& result,
    const Tensor& input,
    double rcond,
    bool hermitian) {
  return torch::linalg_pinv_out(result, input, rcond, hermitian);
}

inline std::tuple<Tensor, Tensor> qr(
    const Tensor& input,
    c10::string_view mode) {
  return torch::linalg_qr(input, mode);
}

inline std::tuple<Tensor&, Tensor&> qr_out(
    Tensor& Q,
    Tensor& R,
    const Tensor& input,
    c10::string_view mode) {
  return torch::linalg_qr_out(Q, R, input, mode);
}

inline std::tuple<Tensor, Tensor> solve_ex(
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return torch::linalg_solve_ex(input, other, left, check_errors);
}

inline std::tuple<Tensor&, Tensor&> solve_ex_out(
    Tensor& result,
    Tensor& info,
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return torch::linalg_solve_ex_out(
      result, info, input, other, left, check_errors);
}

inline Tensor solve(const Tensor& input, const Tensor& other, bool left) {
  return torch::linalg_solve(input, other, left);
}

inline Tensor& solve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool left) {
  return torch::linalg_solve_out(result, input, other, left);
}

inline Tensor solve_triangular(
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return torch::linalg_solve_triangular(
      input, other, upper, left, unitriangular);
}

inline Tensor& solve_triangular_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return torch::linalg_solve_triangular_out(
      result, input, other, upper, left, unitriangular);
}

inline std::tuple<Tensor, Tensor, Tensor> svd(
    const Tensor& input,
    bool full_matrices,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_svd(input, full_matrices, driver);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(
    Tensor& U,
    Tensor& S,
    Tensor& Vh,
    const Tensor& input,
    bool full_matrices,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_svd_out(U, S, Vh, input, full_matrices, driver);
}

inline Tensor svdvals(
    const Tensor& input,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_svdvals(input, driver);
}

inline Tensor& svdvals_out(
    Tensor& result,
    const Tensor& input,
    c10::optional<c10::string_view> driver) {
  return torch::linalg_svdvals_out(result, input, driver);
}

inline Tensor tensorinv(const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv(self, ind);
}

inline Tensor& tensorinv_out(Tensor& result, const Tensor& self, int64_t ind) {
  return torch::linalg_tensorinv_out(result, self, ind);
}

inline Tensor tensorsolve(
    const Tensor& self,
    const Tensor& other,
    OptionalIntArrayRef dims) {
  return torch::linalg_tensorsolve(self, other, dims);
}

inline Tensor& tensorsolve_out(
    Tensor& result,
    const Tensor& self,
    const Tensor& other,
    OptionalIntArrayRef dims) {
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
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.cholesky
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

// C10_DEPRECATED_MESSAGE("linalg_det is deprecated, use det instead.")
inline Tensor linalg_det(const Tensor& self) {
  return detail::det(self);
}

/// See the documentation of torch.linalg.det
inline Tensor det(const Tensor& self) {
  return detail::det(self);
}

/// Computes the sign and (natural) logarithm of the determinant
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.slogdet
inline std::tuple<Tensor, Tensor> slogdet(const Tensor& input) {
  return detail::slogdet(input);
}

inline std::tuple<Tensor&, Tensor&> slogdet_out(
    Tensor& sign,
    Tensor& logabsdet,
    const Tensor& input) {
  return detail::slogdet_out(sign, logabsdet, input);
}

/// Computes eigenvalues and eigenvectors of non-symmetric/non-hermitian
/// matrices
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.eig
inline std::tuple<Tensor, Tensor> eig(const Tensor& self) {
  return detail::eig(self);
}

inline std::tuple<Tensor&, Tensor&> eig_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self) {
  return detail::eig_out(eigvals, eigvecs, self);
}

/// Computes eigenvalues of non-symmetric/non-hermitian matrices
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.eigvals
inline Tensor eigvals(const Tensor& self) {
  return detail::eigvals(self);
}

inline Tensor& eigvals_out(Tensor& result, const Tensor& self) {
  return detail::eigvals_out(result, self);
}

/// Computes eigenvalues and eigenvectors
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.eigh
inline std::tuple<Tensor, Tensor> eigh(
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigh(self, uplo);
}

inline std::tuple<Tensor&, Tensor&> eigh_out(
    Tensor& eigvals,
    Tensor& eigvecs,
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigh_out(eigvals, eigvecs, self, uplo);
}

/// Computes eigenvalues
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.eigvalsh
inline Tensor eigvalsh(const Tensor& self, c10::string_view uplo) {
  return detail::eigvalsh(self, uplo);
}

inline Tensor& eigvalsh_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view uplo) {
  return detail::eigvalsh_out(result, self, uplo);
}

/// Computes the product of Householder matrices
///
/// See
/// https://pytorch.org/docs/main/linalg.html#torch.linalg.householder_product
inline Tensor householder_product(const Tensor& input, const Tensor& tau) {
  return detail::householder_product(input, tau);
}

inline Tensor& householder_product_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tau) {
  return detail::householder_product_out(result, input, tau);
}

inline std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(
    const Tensor& self,
    const Tensor& b,
    c10::optional<double> cond,
    c10::optional<c10::string_view> driver) {
  return detail::lstsq(self, b, cond, driver);
}

/// Computes the matrix exponential
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.matrix_exp
inline Tensor matrix_exp(const Tensor& input) {
  return detail::matrix_exp(input);
}

// C10_DEPRECATED_MESSAGE("linalg_norm is deprecated, use norm instead.")
inline Tensor linalg_norm(
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

// C10_DEPRECATED_MESSAGE("linalg_norm is deprecated, use norm instead.")
inline Tensor linalg_norm(
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);
}

// C10_DEPRECATED_MESSAGE("linalg_norm_out is deprecated, use norm_out
// instead.")
inline Tensor& linalg_norm_out(
    Tensor& result,
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

// C10_DEPRECATED_MESSAGE("linalg_norm_out is deprecated, use norm_out
// instead.")
inline Tensor& linalg_norm_out(
    Tensor& result,
    const Tensor& self,
    c10::string_view ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// Computes the LU factorization with partial pivoting
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.lu_factor
inline std::tuple<Tensor, Tensor> lu_factor(
    const Tensor& input,
    const bool pivot = true) {
  return detail::lu_factor(input, pivot);
}

inline std::tuple<Tensor&, Tensor&> lu_factor_out(
    Tensor& LU,
    Tensor& pivots,
    const Tensor& self,
    const bool pivot = true) {
  return detail::lu_factor_out(LU, pivots, self, pivot);
}

/// Computes the LU factorization with partial pivoting
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.lu
inline std::tuple<Tensor, Tensor, Tensor> lu(
    const Tensor& input,
    const bool pivot = true) {
  return detail::lu(input, pivot);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> lu_out(
    Tensor& P,
    Tensor& L,
    Tensor& U,
    const Tensor& self,
    const bool pivot = true) {
  return detail::lu_out(P, L, U, self, pivot);
}

inline Tensor norm(
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm(self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor norm(
    const Tensor& self,
    std::string ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    const optional<Scalar>& opt_ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, opt_ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& norm_out(
    Tensor& result,
    const Tensor& self,
    std::string ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::norm_out(result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.vector_norm
inline Tensor vector_norm(
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::vector_norm(self, ord, opt_dim, keepdim, opt_dtype);
}

inline Tensor& vector_norm_out(
    Tensor& result,
    const Tensor& self,
    Scalar ord,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype) {
  return detail::vector_norm_out(
      result, self, ord, opt_dim, keepdim, opt_dtype);
}

/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.matrix_norm
inline Tensor matrix_norm(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(
    const Tensor& self,
    const Scalar& ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

inline Tensor matrix_norm(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return detail::matrix_norm(self, ord, dim, keepdim, dtype);
}

inline Tensor& matrix_norm_out(
    const Tensor& self,
    std::string ord,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& result) {
  return detail::matrix_norm_out(self, ord, dim, keepdim, dtype, result);
}

/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.matrix_power
inline Tensor matrix_power(const Tensor& self, int64_t n) {
  return detail::matrix_power(self, n);
}

inline Tensor& matrix_power_out(const Tensor& self, int64_t n, Tensor& result) {
  return detail::matrix_power_out(self, n, result);
}

/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.matrix_rank
inline Tensor matrix_rank(const Tensor& input, double tol, bool hermitian) {
  return detail::matrix_rank(input, tol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return detail::matrix_rank(input, tol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    c10::optional<double> atol,
    c10::optional<double> rtol,
    bool hermitian) {
  return detail::matrix_rank(input, atol, rtol, hermitian);
}

inline Tensor matrix_rank(
    const Tensor& input,
    const c10::optional<Tensor>& atol,
    const c10::optional<Tensor>& rtol,
    bool hermitian) {
  return detail::matrix_rank(input, atol, rtol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    double tol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, tol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& tol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, tol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    c10::optional<double> atol,
    c10::optional<double> rtol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, atol, rtol, hermitian);
}

inline Tensor& matrix_rank_out(
    Tensor& result,
    const Tensor& input,
    const c10::optional<Tensor>& atol,
    const c10::optional<Tensor>& rtol,
    bool hermitian) {
  return detail::matrix_rank_out(result, input, atol, rtol, hermitian);
}

/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.multi_dot
inline Tensor multi_dot(TensorList tensors) {
  return detail::multi_dot(tensors);
}

inline Tensor& multi_dot_out(TensorList tensors, Tensor& result) {
  return detail::multi_dot_out(tensors, result);
}

/// Computes the pseudo-inverse
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.pinv
inline Tensor pinv(
    const Tensor& input,
    double rcond = 1e-15,
    bool hermitian = false) {
  return detail::pinv(input, rcond, hermitian);
}

inline Tensor& pinv_out(
    Tensor& result,
    const Tensor& input,
    double rcond = 1e-15,
    bool hermitian = false) {
  return detail::pinv_out(result, input, rcond, hermitian);
}

/// Computes the QR decomposition
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.qr
inline std::tuple<Tensor, Tensor> qr(
    const Tensor& input,
    c10::string_view mode = "reduced") {
  // C++17 Change the initialisation to "reduced"sv
  //       Same for qr_out
  return detail::qr(input, mode);
}

inline std::tuple<Tensor&, Tensor&> qr_out(
    Tensor& Q,
    Tensor& R,
    const Tensor& input,
    c10::string_view mode = "reduced") {
  return detail::qr_out(Q, R, input, mode);
}

/// Computes the LDL decomposition
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.ldl_factor_ex
inline std::tuple<Tensor, Tensor, Tensor> ldl_factor_ex(
    const Tensor& input,
    bool hermitian,
    bool check_errors) {
  return torch::linalg_ldl_factor_ex(input, hermitian, check_errors);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> ldl_factor_ex_out(
    Tensor& LD,
    Tensor& pivots,
    Tensor& info,
    const Tensor& input,
    bool hermitian,
    bool check_errors) {
  return torch::linalg_ldl_factor_ex_out(
      LD, pivots, info, input, hermitian, check_errors);
}

/// Solve a system of linear equations using the LDL decomposition
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.ldl_solve
inline Tensor ldl_solve(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool hermitian) {
  return torch::linalg_ldl_solve(LD, pivots, B, hermitian);
}

inline Tensor& ldl_solve_out(
    Tensor& result,
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool hermitian) {
  return torch::linalg_ldl_solve_out(result, LD, pivots, B, hermitian);
}

/// Solves a system linear system AX = B
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.solve_ex
inline std::tuple<Tensor, Tensor> solve_ex(
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return detail::solve_ex(input, other, left, check_errors);
}

inline std::tuple<Tensor&, Tensor&> solve_ex_out(
    Tensor& result,
    Tensor& info,
    const Tensor& input,
    const Tensor& other,
    bool left,
    bool check_errors) {
  return detail::solve_ex_out(result, info, input, other, left, check_errors);
}

/// Computes a tensor `x` such that `matmul(input, x) = other`.
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.solve
inline Tensor solve(const Tensor& input, const Tensor& other, bool left) {
  return detail::solve(input, other, left);
}

inline Tensor& solve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool left) {
  return detail::solve_out(result, input, other, left);
}

/// Computes a solution of a linear system AX = B for input = A and other = B
/// whenever A is square upper or lower triangular and does not have zeros in
/// the diagonal
///
/// See
/// https://pytorch.org/docs/main/linalg.html#torch.linalg.solve_triangular
inline Tensor solve_triangular(
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return detail::solve_triangular(input, other, upper, left, unitriangular);
}

inline Tensor& solve_triangular_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    bool upper,
    bool left,
    bool unitriangular) {
  return detail::solve_triangular_out(
      result, input, other, upper, left, unitriangular);
}

/// Computes the singular values and singular vectors
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.svd
inline std::tuple<Tensor, Tensor, Tensor> svd(
    const Tensor& input,
    bool full_matrices,
    c10::optional<c10::string_view> driver) {
  return detail::svd(input, full_matrices, driver);
}

inline std::tuple<Tensor&, Tensor&, Tensor&> svd_out(
    Tensor& U,
    Tensor& S,
    Tensor& Vh,
    const Tensor& input,
    bool full_matrices,
    c10::optional<c10::string_view> driver) {
  return detail::svd_out(U, S, Vh, input, full_matrices, driver);
}

/// Computes the singular values
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.svdvals
inline Tensor svdvals(
    const Tensor& input,
    c10::optional<c10::string_view> driver) {
  return detail::svdvals(input, driver);
}

inline Tensor& svdvals_out(
    Tensor& result,
    const Tensor& input,
    c10::optional<c10::string_view> driver) {
  return detail::svdvals_out(result, input, driver);
}

/// Computes the inverse of a tensor
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.tensorinv
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
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.tensorsolve
///
/// Example:
/// ```
/// auto a = torch::eye(2*3*4).reshape({2*3, 4, 2, 3, 4});
/// auto b = torch::randn(2*3, 4);
/// auto x = torch::linalg::tensorsolve(a, b);
/// ```
inline Tensor tensorsolve(
    const Tensor& input,
    const Tensor& other,
    OptionalIntArrayRef dims) {
  return detail::tensorsolve(input, other, dims);
}

inline Tensor& tensorsolve_out(
    Tensor& result,
    const Tensor& input,
    const Tensor& other,
    OptionalIntArrayRef dims) {
  return detail::tensorsolve_out(result, input, other, dims);
}

/// Computes a tensor `inverse_input` such that `dot(input, inverse_input) =
/// eye(input.size(0))`.
///
/// See https://pytorch.org/docs/main/linalg.html#torch.linalg.inv
inline Tensor inv(const Tensor& input) {
  return detail::inv(input);
}

inline Tensor& inv_out(Tensor& result, const Tensor& input) {
  return detail::inv_out(result, input);
}

} // namespace linalg
} // namespace torch
