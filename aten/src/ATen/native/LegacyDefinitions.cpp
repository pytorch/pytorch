#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

namespace at { namespace native {

// Methods

void* data_ptr(const Tensor & self) {
  return self.unsafeGetTensorImpl()->slow_data();
}

Tensor & set_(Tensor& self, Storage source) {
  return at::_th_set_(self, source);
}

Tensor & set_(Tensor& self, Storage source, int64_t storage_offset, IntList size, IntList stride) {
  return at::_th_set_(self, source, storage_offset, size, stride);
}

Tensor & set_(Tensor& self, const Tensor & source) {
  return at::_th_set_(self, source);
}

Tensor & set_(Tensor& self) {
  return at::_th_set_(self);
}

bool is_set_to(const Tensor& self, const Tensor & tensor) {
  return at::_th_is_set_to(self, tensor);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, Scalar value) {
  return at::_th_masked_fill_(self, mask, value);
}

Tensor & masked_fill_(Tensor& self, const Tensor & mask, const Tensor & value) {
  return at::_th_masked_fill_(self, mask, value);
}

Tensor & masked_scatter_(Tensor& self, const Tensor & mask, const Tensor & source) {
  return at::_th_masked_scatter_(self, mask, source);
}

Tensor view(const Tensor& self, IntList size) {
  return at::_th_view(self, size);
}

Tensor & put_(Tensor& self, const Tensor & index, const Tensor & source, bool accumulate) {
  return at::_th_put_(self, index, source, accumulate);
}

Tensor & index_add_(Tensor& self, int64_t dim, const Tensor & index, const Tensor & source) {
  return at::_th_index_add_(self, dim, index, source);
}

Tensor & index_fill_(Tensor& self, int64_t dim, const Tensor & index, Scalar value) {
  return at::_th_index_fill_(self, dim, index, value);
}

Tensor & index_fill_(Tensor& self, int64_t dim, const Tensor & index, const Tensor & value) {
  return at::_th_index_fill_(self, dim, index, value);
}

Tensor & scatter_(Tensor& self, int64_t dim, const Tensor & index, const Tensor & src) {
  return at::_th_scatter_(self, dim, index, src);
}

Tensor & scatter_(Tensor& self, int64_t dim, const Tensor & index, Scalar value) {
  return at::_th_scatter_(self, dim, index, value);
}

Tensor & scatter_add_(Tensor& self, int64_t dim, const Tensor & index, const Tensor & src) {
  return at::_th_scatter_add_(self, dim, index, src);
}

Tensor & lt_(Tensor& self, Scalar other) {
  return at::_th_lt_(self, other);
}

Tensor & lt_(Tensor& self, const Tensor & other) {
  return at::_th_lt_(self, other);
}

Tensor & gt_(Tensor& self, Scalar other) {
  return at::_th_gt_(self, other);
}

Tensor & gt_(Tensor& self, const Tensor & other) {
  return at::_th_gt_(self, other);
}

Tensor & le_(Tensor& self, Scalar other) {
  return at::_th_le_(self, other);
}

Tensor & le_(Tensor& self, const Tensor & other) {
  return at::_th_le_(self, other);
}

Tensor & ge_(Tensor& self, Scalar other) {
  return at::_th_ge_(self, other);
}

Tensor & ge_(Tensor& self, const Tensor & other) {
  return at::_th_ge_(self, other);
}

Tensor & eq_(Tensor& self, Scalar other) {
  return at::_th_eq_(self, other);
}

Tensor & eq_(Tensor& self, const Tensor & other) {
  return at::_th_ge_(self, other);
}

Tensor & ne_(Tensor& self, Scalar other) {
  return at::_th_ne_(self, other);
}

Tensor & ne_(Tensor& self, const Tensor & other) {
  return at::_th_ne_(self, other);
}

Tensor & lgamma_(Tensor& self) {
  return at::_th_lgamma_(self);
}

Tensor & atan2_(Tensor& self, const Tensor & other) {
  return at::_th_atan2_(self, other);
}

Tensor & tril_(Tensor& self, int64_t diagonal) {
  return at::_th_tril_(self, diagonal);
}

Tensor & triu_(Tensor& self, int64_t diagonal) {
  return self._th_triu_(diagonal);
}

Tensor & digamma_(Tensor& self) {
  return at::_th_digamma_(self);
}

Tensor & polygamma_(Tensor& self, int64_t n) {
  return at::_th_polygamma_(self, n);
}

Tensor & erfinv_(Tensor& self) {
  return at::_th_erfinv_(self);
}

Tensor & frac_(Tensor& self) {
  return at::_th_frac_(self);
}

Tensor & renorm_(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::_th_renorm_(self, p, dim, maxnorm);
}

Tensor & reciprocal_(Tensor& self) {
  return at::_th_reciprocal_(self);
}

Tensor & neg_(Tensor& self) {
  return at::_th_neg_(self);
}

Tensor & pow_(Tensor& self, Scalar exponent) {
  return at::_th_pow_(self, exponent);
}

Tensor & pow_(Tensor& self, const Tensor & exponent) {
  return at::_th_pow_(self, exponent);
}

Tensor & lerp_(Tensor& self, const Tensor & end, Scalar weight) {
  return at::_th_lerp_(self, end, weight);
}

Tensor & sign_(Tensor& self) {
  return at::_th_sign_(self);
}

Tensor & fmod_(Tensor& self, Scalar other) {
  return at::_th_fmod_(self, other);
}

Tensor & fmod_(Tensor& self, const Tensor & other) {
  return at::_th_fmod_(self, other);
}

Tensor & remainder_(Tensor& self, Scalar other) {
  return at::_th_remainder_(self, other);
}

Tensor & remainder_(Tensor& self, const Tensor & other) {
  return at::_th_remainder_(self, other);
}

Tensor & addbmm_(Tensor& self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::_th_addbmm_(self, batch1, batch2, beta, alpha);
}

Tensor & addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::_th_addbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::_th_addbmm(self, batch1, batch2, beta, alpha);
}

Tensor & addcmul_(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcmul_(self, tensor1, tensor2, value);
}

Tensor & addcdiv_(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcdiv_(self, tensor1, tensor2, value);
}

Tensor & random_(Tensor& self, int64_t from, int64_t to, Generator * generator) {
  return at::_th_random_(self, from, to, generator);
}

Tensor & random_(Tensor& self, int64_t to, Generator * generator) {
  return at::_th_random_(self, to, generator);
}

Tensor & random_(Tensor& self, Generator * generator) {
  return at::_th_random_(self, generator);
}

Tensor & uniform_(Tensor& self, double from, double to, Generator * generator) {
  return at::_th_uniform_(self, from, to, generator);
}

Tensor & normal_(Tensor& self, double mean, double std, Generator * generator) {
  return at::_th_normal_(self, mean, std, generator);
}

Tensor & cauchy_(Tensor& self, double median, double sigma, Generator * generator) {
  return at::_th_cauchy_(self, median, sigma, generator);
}

Tensor & log_normal_(Tensor& self, double mean, double std, Generator * generator) {
  return at::_th_log_normal_(self, mean, std, generator);
}

Tensor & exponential_(Tensor& self, double lambd, Generator * generator) {
  return at::_th_exponential_(self, lambd, generator);
}

Tensor & geometric_(Tensor& self, double p, Generator * generator) {
  return at::_th_geometric_(self, p, generator);
}

// Functions

Tensor & diag_out(Tensor & result, const Tensor & self, int64_t diagonal) {
  return at::_th_diag_out(result, self, diagonal);
}

Tensor diag(const Tensor & self, int64_t diagonal) {
  return at::_th_diag(self, diagonal);
}

Tensor & cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) {
  return at::_th_cross_out(result, self, other, dim);
}

Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) {
  return at::_th_cross(self, other, dim);
}

Tensor & triu_out(Tensor & result, const Tensor & self, int64_t diagonal) {
  return at::_th_triu_out(result, self, diagonal);
}

Tensor triu(const Tensor & self, int64_t diagonal) {
  return at::_th_triu(self, diagonal);
}

Tensor & tril_out(Tensor & result, const Tensor & self, int64_t diagonal) {
  return at::_th_tril_out(result, self, diagonal);
}

Tensor tril(const Tensor & self, int64_t diagonal) {
  return at::_th_tril(self, diagonal);
}

Tensor trace(const Tensor & self) {
  return at::_th_trace(self);
}

Tensor & ne_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_ne_out(result, self, other);
}

Tensor ne(const Tensor & self, Scalar other) {
  return at::_th_ne(self, other);
}

Tensor & ne_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_ne_out(result, self, other);
}

Tensor ne(const Tensor & self, const Tensor & other) {
  return at::_th_ne(self, other);
}

Tensor & eq_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_eq_out(result, self, other);
}

Tensor eq(const Tensor & self, Scalar other) {
  return at::_th_eq(self, other);
}

Tensor & eq_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_eq_out(result, self, other);
}

Tensor eq(const Tensor & self, const Tensor & other) {
  return at::_th_eq(self, other);
}

Tensor & ge_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_ge_out(result, self, other);
}

Tensor ge(const Tensor & self, Scalar other) {
  return at::_th_ge(self, other);
}

Tensor & ge_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_ge_out(result, self, other);
}

Tensor ge(const Tensor & self, const Tensor & other) {
  return at::_th_ge(self, other);
}

Tensor & le_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_le_out(result, self, other);
}

Tensor le(const Tensor & self, Scalar other) {
  return at::_th_le(self, other);
}

Tensor & le_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_le_out(result, self, other);
}

Tensor le(const Tensor & self, const Tensor & other) {
  return at::_th_le(self, other);
}

Tensor & gt_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_gt_out(result, self, other);
}

Tensor gt(const Tensor & self, Scalar other) {
  return at::_th_gt(self, other);
}

Tensor & gt_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_gt_out(result, self, other);
}

Tensor gt(const Tensor & self, const Tensor & other) {
  return at::_th_gt(self, other);
}

Tensor & lt_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_lt_out(result, self, other);
}

Tensor lt(const Tensor & self, Scalar other) {
  return at::_th_lt(self, other);
}

Tensor & lt_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_lt_out(result, self, other);
}

Tensor lt(const Tensor & self, const Tensor & other) {
  return at::_th_lt(self, other);
}

Tensor & take_out(Tensor & result, const Tensor & self, const Tensor & index) {
  return at::_th_take_out(result, self, index);
}

Tensor take(const Tensor & self, const Tensor & index) {
  return at::_th_take(self, index);
}

Tensor & index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  return at::_th_index_select_out(result, self, dim, index);
}

Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
  return at::_th_index_select(self, dim, index);
}

Tensor & masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) {
  return at::_th_masked_select_out(result, self, mask);
}

Tensor masked_select(const Tensor & self, const Tensor & mask) {
  return at::_th_masked_select(self, mask);
}

Tensor & nonzero_out(Tensor & result, const Tensor & self) {
  return at::_th_nonzero_out(result, self);
}

Tensor nonzero(const Tensor & self) {
  return at::_th_nonzero(self);
}

Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  return at::_th_gather_out(result, self, dim, index);
}

Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) {
  return at::_th_gather(self, dim, index);
}

Tensor & addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcmul_out(result, self, tensor1, tensor2, value);
}

Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcmul(self, tensor1, tensor2, value);
}

Tensor & addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcdiv_out(result, self, tensor1, tensor2, value);
}

Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::_th_addcdiv(self, tensor1, tensor2, value);
}

std::tuple<Tensor &,Tensor &> gels_out(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) {
  return at::_th_gels_out(X, qr, self, A);
}

std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) {
  return at::_th_gels(self, A);
}

std::tuple<Tensor &,Tensor &> trtrs_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  return at::_th_trtrs_out(X, M, self, A, upper, transpose, unitriangular);
}

std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
  return at::_th_trtrs(self, A, upper, transpose, unitriangular);
}

std::tuple<Tensor &,Tensor &> symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) {
  return at::_th_symeig_out(e, V, self, eigenvectors, upper);
}

std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
  return at::_th_symeig(self, eigenvectors, upper);
}

std::tuple<Tensor &,Tensor &> eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) {
  return at::_th_eig_out(e, v, self, eigenvectors);
}

std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
  return at::_th_eig(self, eigenvectors);
}

std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) {
  return at::_th_svd_out(U, S, V, self, some, compute_uv);
}

std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv) {
  return at::_th_svd(self, some, compute_uv);
}

Tensor & cholesky_out(Tensor & result, const Tensor & self, bool upper) {
  return at::_th_potrf_out(result, self, upper);
}

Tensor cholesky(const Tensor & self, bool upper) {
  return at::_th_potrf(self, upper);
}

Tensor & potri_out(Tensor & result, const Tensor & self, bool upper) {
  return at::_th_potri_out(result, self, upper);
}

Tensor potri(const Tensor & self, bool upper) {
  return at::_th_potri(self, upper);
}

std::tuple<Tensor &,Tensor &> pstrf_out(Tensor & u, Tensor & piv, const Tensor & self, bool upper, Scalar tol) {
  return at::_th_pstrf_out(u, piv, self, upper, tol);
}

std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) {
  return at::_th_pstrf(self, upper, tol);
}

std::tuple<Tensor &,Tensor &> qr_out(Tensor & Q, Tensor & R, const Tensor & self) {
  return at::_th_qr_out(Q, R, self);
}

std::tuple<Tensor,Tensor> qr(const Tensor & self) {
  return at::_th_qr(self);
}

std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & result0, Tensor & result1, const Tensor & self) {
  return at::geqrf_out(result0, result1, self);
}

std::tuple<Tensor,Tensor> geqrf(const Tensor & self) {
  return at::_th_geqrf(self);
}

Tensor & orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) {
  return at::_th_orgqr_out(result, self, input2);
}

Tensor orgqr(const Tensor & self, const Tensor & input2) {
  return at::_th_orgqr(self, input2);
}

Tensor & ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  return at::ormqr_out(result, self, input2, input3, left, transpose);
}

Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
  return at::_th_ormqr(self, input2, input3, left, transpose);
}

std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & A_LU, Tensor & pivots, const Tensor & self, bool pivot) {
  return at::_th_btrifact_out(A_LU, pivots, self, pivot);
}

std::tuple<Tensor,Tensor> btrifact(const Tensor & self, bool pivot) {
  return at::_th_btrifact(self, pivot);
}

std::tuple<Tensor &,Tensor &,Tensor &> btrifact_with_info_out(Tensor & A_LU, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) {
  return at::_th_btrifact_with_info_out(A_LU, pivots, info, self, pivot);
}

std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(const Tensor & self, bool pivot) {
  return at::_th_btrifact_with_info(self, pivot);
}

Tensor & btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  return at::_th_btrisolve_out(result, self, LU_data, LU_pivots);
}

Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  return at::_th_btrisolve(self, LU_data, LU_pivots);
}

Tensor & multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
  return at::_th_multinomial_out(result, self, num_samples, replacement, generator);
}

Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
  return at::_th_multinomial(self, num_samples, replacement, generator);
}

Tensor & lgamma_out(Tensor & result, const Tensor & self) {
  return at::_th_lgamma_out(result, self);
}

Tensor lgamma(const Tensor & self) {
  return at::_th_lgamma(self);
}

Tensor & digamma_out(Tensor & result, const Tensor & self) {
  return at::_th_digamma_out(result, self);
}
Tensor digamma(const Tensor & self) {
  return at::_th_digamma(self);
}

Tensor & polygamma_out(Tensor & result, int64_t n, const Tensor & self) {
  return at::_th_polygamma_out(result, n, self);
}

Tensor polygamma(int64_t n, const Tensor & self) {
  return at::_th_polygamma(n, self);
}

Tensor & erfinv_out(Tensor & result, const Tensor & self) {
  return at::_th_erfinv_out(result, self);
}

Tensor erfinv(const Tensor & self) {
  return at::_th_erfinv(self);
}

Tensor & frac_out(Tensor & result, const Tensor & self) {
  return at::_th_frac_out(result, self);
}

Tensor frac(const Tensor & self) {
  return at::_th_frac(self);
}

Tensor dist(const Tensor & self, const Tensor & other, Scalar p) {
  return at::_th_dist(self, other, p);
}

Tensor & reciprocal_out(Tensor & result, const Tensor & self) {
  return at::_th_reciprocal_out(result, self);
}

Tensor reciprocal(const Tensor & self) {
  return at::_th_reciprocal(self);
}

Tensor & neg_out(Tensor & result, const Tensor & self) {
  return at::_th_neg_out(result, self);
}

Tensor neg(const Tensor & self) {
  return at::_th_neg(self);
}

Tensor & atan2_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_atan2_out(result, self, other);
}

Tensor atan2(const Tensor & self, const Tensor & other) {
  return at::_th_atan2(self, other);
}

Tensor & lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) {
  return at::_th_lerp_out(result, self, end, weight);
}

Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) {
  return at::_th_lerp(self, end, weight);
}

Tensor & histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  return at::_th_histc_out(result, self, bins, min, max);
}

Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
  return at::_th_histc(self, bins, min, max);
}

Tensor & sign_out(Tensor & result, const Tensor & self) {
  return at::_th_sign_out(result, self);
}

Tensor sign(const Tensor & self) {
  return at::_th_sign(self);
}

Tensor & fmod_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_fmod_out(result, self, other);
}

Tensor fmod(const Tensor & self, Scalar other) {
  return at::_th_fmod(self, other);
}

Tensor & fmod_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_fmod_out(result, self, other);
}

Tensor fmod(const Tensor & self, const Tensor & other) {
  return at::_th_fmod(self, other);
}

Tensor & remainder_out(Tensor & result, const Tensor & self, Scalar other) {
  return at::_th_remainder_out(result, self, other);
}

Tensor remainder(const Tensor & self, Scalar other) {
  return at::_th_remainder(self, other);
}

Tensor & remainder_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_remainder_out(result, self, other);
}

Tensor remainder(const Tensor & self, const Tensor & other) {
  return at::_th_remainder(self, other);
}

Tensor & min_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_min_out(result, self, other);
}

Tensor min(const Tensor & self, const Tensor & other) {
  return at::_th_min(self, other);
}

Tensor min(const Tensor & self) {
  return at::_th_min(self);
}

Tensor & max_out(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::_th_max_out(result, self, other);
}
Tensor max(const Tensor & self, const Tensor & other) {
  return at::_th_max(self, other);
}

Tensor max(const Tensor & self) {
  return at::_th_max(self);
}

Tensor median(const Tensor & self) {
  return at::_th_median(self);
}

std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  return at::_th_sort_out(values, indices, self, dim, descending);
}

std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
  return at::_th_sort(self, dim, descending);
}
std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  return at::_th_topk_out(values, indices, self, k, dim, largest, sorted);
}

std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  return at::_th_topk(self, k, dim, largest, sorted);
}

Tensor all(const Tensor & self) {
  return at::_th_all(self);
}

Tensor any(const Tensor & self) {
  return at::_th_any(self);
}

Tensor & renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::_th_renorm_out(result, self, p, dim, maxnorm);
}

Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::_th_renorm(self, p, dim, maxnorm);
}

Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  return at::_th_unfold(self, dimension, size, step);
}

bool equal(const Tensor & self, const Tensor & other) {
  return at::_th_equal(self, other);
}

Tensor & pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) {
  return at::_th_pow_out(result, self, exponent);
}

Tensor pow(const Tensor & self, const Tensor & exponent) {
  return at::_th_pow(self, exponent);
}
Tensor & pow_out(Tensor & result, Scalar self, const Tensor & exponent) {
  return at::_th_pow_out(result, self, exponent);
}

Tensor pow(Scalar self, const Tensor & exponent) {
  return at::_th_pow(self, exponent);
}

Tensor & normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) {
  return at::_th_normal_out(output, mean, std, generator);
}

Tensor normal(const Tensor & mean, double std, Generator * generator) {
  return at::_th_normal(mean, std, generator);
}

Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) {
  return at::_th_normal_out(output, mean, std, generator);
}

Tensor normal(double mean, const Tensor & std, Generator * generator) {
  return at::_th_normal(mean, std, generator);
}

Tensor & normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) {
  return at::_th_normal_out(output, mean, std, generator);
}

Tensor normal(const Tensor & mean, const Tensor & std, Generator * generator) {
  return at::_th_normal(mean, std, generator);
}

Tensor alias(const Tensor & self) {
  return at::_th_alias(self);
}

Tensor & _dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) {
  return at::_th_dirichlet_grad_out(output, x, alpha, total);
}

Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {
  return at::_th_dirichlet_grad(x, alpha, total);
}

Tensor __and__(const Tensor & self, Scalar other) {
  return at::_th_and(self, other);
}

Tensor __and__(const Tensor & self, const Tensor & other) {
  return at::_th_and(self, other);
}

Tensor __or__(const Tensor & self, Scalar other) {
  return at::_th_or(self, other);
}

Tensor __or__(const Tensor & self, const Tensor & other) {
  return at::_th_or(self, other);
}

Tensor __xor__(const Tensor & self, Scalar other) {
  return at::_th_xor(self, other);
}

Tensor __xor__(const Tensor & self, const Tensor & other) {
  return at::_th_xor(self, other);
}

Tensor __lshift__(const Tensor & self, Scalar other) {
  return at::_th_lshift(self, other);
}

Tensor __lshift__(const Tensor & self, const Tensor & other) {
  return at::_th_lshift(self, other);
}

Tensor __rshift__(const Tensor & self, Scalar other) {
  return at::_th_rshift(self, other);
}

Tensor __rshift__(const Tensor & self, const Tensor & other) {
  return at::_th_rshift(self, other);
}

Tensor & __iand__(Tensor & self, Scalar other) {
  return at::_th_iand_(self, other);
}

Tensor & __iand__(Tensor & self, const Tensor & other) {
  return at::_th_iand_(self, other);
}

Tensor & __ior__(Tensor & self, Scalar other) {
  return at::_th_ior_(self, other);
}

Tensor & __ior__(Tensor & self, const Tensor & other) {
  return at::_th_ior_(self, other);
}

Tensor & __ixor__(Tensor & self, Scalar other) {
  return at::_th_ixor_(self, other);
}

Tensor & __ixor__(Tensor & self, const Tensor & other) {
  return at::_th_ixor_(self, other);
}

Tensor & __ilshift__(Tensor & self, Scalar other) {
  return at::_th_ilshift_(self, other);
}

Tensor & __ilshift__(Tensor & self, const Tensor & other) {
  return at::_th_ilshift_(self, other);
}

Tensor & __irshift__(Tensor & self, Scalar other) {
  return at::_th_irshift_(self, other);
}

Tensor & __irshift__(Tensor & self, const Tensor & other) {
  return at::_th_irshift_(self, other);
}

}} // namespace at::native
