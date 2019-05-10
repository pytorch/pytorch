#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCUDA.h>

namespace at { namespace native {

// Methods

void* data_ptr(const Tensor & self) {
  return self.unsafeGetTensorImpl()->slow_data();
}

Tensor & set__cuda(Tensor& self, Storage source) {
  return at::legacy::cuda::_th_set_(self, source);
}

Tensor & set__cuda(Tensor& self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  return at::legacy::cuda::_th_set_(self, source, storage_offset, size, stride);
}

Tensor & set__cuda(Tensor& self, const Tensor & source) {
  return at::legacy::cuda::_th_set_(self, source);
}

Tensor & set__cuda(Tensor& self) {
  return at::legacy::cuda::_th_set_(self);
}

bool is_set_to_cuda(const Tensor& self, const Tensor & tensor) {
  return at::legacy::cuda::_th_is_set_to(self, tensor);
}

Tensor clone_cuda(const Tensor& self) {
  return at::legacy::cuda::_th_clone(self);
}

Tensor& resize_as__cuda(Tensor& self, const Tensor& the_template) {
  return at::legacy::cuda::_th_resize_as_(self, the_template);
}

Tensor& pow_out_cuda(Tensor& result, const Tensor& self, Scalar exponent) {
  return at::legacy::cuda::_th_pow_out(result, self, exponent);
}

Tensor pow_cuda(const Tensor& self, Scalar exponent) {
  return at::legacy::cuda::_th_pow(self, exponent);
}

Tensor& zero__cuda(Tensor& self) {
  return at::legacy::cuda::_th_zero_(self);
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, Scalar value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return at::legacy::cuda::_th_masked_fill_(self, mask, value);
  } else {
    return at::legacy::cuda::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_fill__cuda(Tensor& self, const Tensor & mask, const Tensor & value) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return at::legacy::cuda::_th_masked_fill_(self, mask, value);
  } else {
    return at::legacy::cuda::_th_masked_fill_bool_(self, mask, value);
  }
}

Tensor & masked_scatter__cuda(Tensor& self, const Tensor & mask, const Tensor & source) {
  // As we dispatch on self and TH is type-checked, we need different definitions.
  // This can be fixed by moving to ATen.
  if (mask.dtype() == at::ScalarType::Byte) {
    return at::legacy::cuda::_th_masked_scatter_(self, mask, source);
  } else {
    return at::legacy::cuda::_th_masked_scatter_bool_(self, mask, source);
  }
}

Tensor view_cuda(const Tensor& self, IntArrayRef size) {
  return at::legacy::cuda::_th_view(self, size);
}

Tensor & put__cuda(Tensor& self, const Tensor & index, const Tensor & source, bool accumulate) {
  return at::legacy::cuda::_th_put_(self, index, source, accumulate);
}

Tensor & index_add__cuda(Tensor& self, int64_t dim, const Tensor & index, const Tensor & source) {
  return at::legacy::cuda::_th_index_add_(self, dim, index, source);
}

Tensor & index_fill__cuda(Tensor& self, int64_t dim, const Tensor & index, Scalar value) {
  return at::legacy::cuda::_th_index_fill_(self, dim, index, value);
}

Tensor & index_fill__cuda(Tensor& self, int64_t dim, const Tensor & index, const Tensor & value) {
  return at::legacy::cuda::_th_index_fill_(self, dim, index, value);
}

Tensor & scatter__cuda(Tensor& self, int64_t dim, const Tensor & index, const Tensor & src) {
  return at::legacy::cuda::_th_scatter_(self, dim, index, src);
}

Tensor & scatter__cuda(Tensor& self, int64_t dim, const Tensor & index, Scalar value) {
  return at::legacy::cuda::_th_scatter_(self, dim, index, value);
}

Tensor & scatter_add__cuda(Tensor& self, int64_t dim, const Tensor & index, const Tensor & src) {
  return at::legacy::cuda::_th_scatter_add_(self, dim, index, src);
}

Tensor & lt__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_lt_(self, other);
}

Tensor & lt__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_lt_(self, other);
}

Tensor & gt__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_gt_(self, other);
}

Tensor & gt__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_gt_(self, other);
}

Tensor & le__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_le_(self, other);
}

Tensor & le__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_le_(self, other);
}

Tensor & ge__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_ge_(self, other);
}

Tensor & ge__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_ge_(self, other);
}

Tensor & eq__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_eq_(self, other);
}

Tensor & eq__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_eq_(self, other);
}

Tensor & ne__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_ne_(self, other);
}

Tensor & ne__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_ne_(self, other);
}

Tensor & lgamma__cuda(Tensor& self) {
  return at::legacy::cuda::_th_lgamma_(self);
}

Tensor & atan2__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_atan2_(self, other);
}

Tensor & digamma__cuda(Tensor& self) {
  return at::legacy::cuda::_th_digamma_(self);
}

Tensor & polygamma__cuda(Tensor& self, int64_t n) {
  return at::legacy::cuda::_th_polygamma_(self, n);
}

Tensor & erfinv__cuda(Tensor& self) {
  return at::legacy::cuda::_th_erfinv_(self);
}

Tensor & renorm__cuda(Tensor& self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::legacy::cuda::_th_renorm_(self, p, dim, maxnorm);
}

Tensor & pow__cuda(Tensor& self, Scalar exponent) {
  return at::legacy::cuda::_th_pow_(self, exponent);
}

Tensor & pow__cuda(Tensor& self, const Tensor & exponent) {
  return at::legacy::cuda::_th_pow_(self, exponent);
}

Tensor & sign__cuda(Tensor& self) {
  return at::legacy::cuda::_th_sign_(self);
}

Tensor & fmod__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_fmod_(self, other);
}

Tensor & fmod__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_fmod_(self, other);
}

Tensor & remainder__cuda(Tensor& self, Scalar other) {
  return at::legacy::cuda::_th_remainder_(self, other);
}

Tensor & remainder__cuda(Tensor& self, const Tensor & other) {
  return at::legacy::cuda::_th_remainder_(self, other);
}

Tensor & addbmm__cuda(Tensor& self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addbmm_(self, batch1, batch2, beta, alpha);
}

Tensor & addbmm_out_cuda(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addbmm_out(result, self, batch1, batch2, beta, alpha);
}

Tensor addbmm_cuda(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addbmm(self, batch1, batch2, beta, alpha);
}

Tensor & addcmul__cuda(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcmul_(self, tensor1, tensor2, value);
}

Tensor & addcdiv__cuda(Tensor& self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcdiv_(self, tensor1, tensor2, value);
}

Tensor & random__cuda(Tensor& self, int64_t from, int64_t to, Generator * generator) {
  return at::legacy::cuda::_th_random_(self, from, to, generator);
}

Tensor & random__cuda(Tensor& self, int64_t to, Generator * generator) {
  return at::legacy::cuda::_th_random_(self, to, generator);
}

Tensor & random__cuda(Tensor& self, Generator * generator) {
  return at::legacy::cuda::_th_random_(self, generator);
}

Tensor & uniform__cuda(Tensor& self, double from, double to, Generator * generator) {
  return at::legacy::cuda::_th_uniform_(self, from, to, generator);
}

Tensor & normal__cuda(Tensor& self, double mean, double std, Generator * generator) {
  return at::legacy::cuda::_th_normal_(self, mean, std, generator);
}

Tensor & cauchy__cuda(Tensor& self, double median, double sigma, Generator * generator) {
  return at::legacy::cuda::_th_cauchy_(self, median, sigma, generator);
}

Tensor & log_normal__cuda(Tensor& self, double mean, double std, Generator * generator) {
  return at::legacy::cuda::_th_log_normal_(self, mean, std, generator);
}

Tensor & exponential__cuda(Tensor& self, double lambd, Generator * generator) {
  return at::legacy::cuda::_th_exponential_(self, lambd, generator);
}

Tensor & geometric__cuda(Tensor& self, double p, Generator * generator) {
  return at::legacy::cuda::_th_geometric_(self, p, generator);
}

// Functions

Tensor & diag_out_cuda(Tensor & result, const Tensor & self, int64_t diagonal) {
  return at::legacy::cuda::_th_diag_out(result, self, diagonal);
}

Tensor diag_cuda(const Tensor & self, int64_t diagonal) {
  return at::legacy::cuda::_th_diag(self, diagonal);
}

Tensor trace_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_trace(self);
}

Tensor & ne_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ne_out(result, self, other);
}

Tensor ne_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ne(self, other);
}

Tensor & ne_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ne_out(result, self, other);
}

Tensor ne_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ne(self, other);
}

Tensor & eq_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_eq_out(result, self, other);
}

Tensor eq_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_eq(self, other);
}

Tensor & eq_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_eq_out(result, self, other);
}

Tensor eq_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_eq(self, other);
}

Tensor & ge_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ge_out(result, self, other);
}

Tensor ge_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ge(self, other);
}

Tensor & ge_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ge_out(result, self, other);
}

Tensor ge_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ge(self, other);
}

Tensor & le_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_le_out(result, self, other);
}

Tensor le_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_le(self, other);
}

Tensor & le_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_le_out(result, self, other);
}

Tensor le_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_le(self, other);
}

Tensor & gt_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_gt_out(result, self, other);
}

Tensor gt_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_gt(self, other);
}

Tensor & gt_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_gt_out(result, self, other);
}

Tensor gt_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_gt(self, other);
}

Tensor & lt_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_lt_out(result, self, other);
}

Tensor lt_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_lt(self, other);
}

Tensor & lt_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_lt_out(result, self, other);
}

Tensor lt_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_lt(self, other);
}

Tensor & take_out_cuda(Tensor & result, const Tensor & self, const Tensor & index) {
  return at::legacy::cuda::_th_take_out(result, self, index);
}

Tensor take_cuda(const Tensor & self, const Tensor & index) {
  return at::legacy::cuda::_th_take(self, index);
}

Tensor & index_select_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
  return at::legacy::cuda::_th_index_select_out(result, self, dim, index);
}

Tensor index_select_cuda(const Tensor & self, int64_t dim, const Tensor & index) {
  return at::legacy::cuda::_th_index_select(self, dim, index);
}

Tensor & masked_select_out_cuda(Tensor & result, const Tensor & self, const Tensor & mask) {
  return at::legacy::cuda::_th_masked_select_out(result, self, mask);
}

Tensor masked_select_cuda(const Tensor & self, const Tensor & mask) {
  if (mask.dtype() == at::ScalarType::Byte) {
  return at::legacy::cuda::_th_masked_select(self, mask);
} else {
  return at::legacy::cuda::_th_masked_select_bool(self, mask);
}
}

Tensor & nonzero_out_cuda(Tensor & result, const Tensor & self) {
  return at::legacy::cuda::_th_nonzero_out(result, self);
}

Tensor nonzero_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_nonzero(self);
}

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return at::legacy::cuda::_th_gather_out(result, self, dim, index);
}

Tensor gather_cuda(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return at::legacy::cuda::_th_gather(self, dim, index);
}

Tensor & addcmul_out_cuda(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcmul_out(result, self, tensor1, tensor2, value);
}

Tensor addcmul_cuda(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcmul(self, tensor1, tensor2, value);
}

Tensor & addcdiv_out_cuda(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcdiv_out(result, self, tensor1, tensor2, value);
}

Tensor addcdiv_cuda(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
  return at::legacy::cuda::_th_addcdiv(self, tensor1, tensor2, value);
}

std::tuple<Tensor &,Tensor &> gels_out_cuda(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A) {
  return at::legacy::cuda::_th_gels_out(X, qr, self, A);
}

std::tuple<Tensor,Tensor> gels_cuda(const Tensor & self, const Tensor & A) {
  return at::legacy::cuda::_th_gels(self, A);
}

std::tuple<Tensor &,Tensor &> symeig_out_cuda(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper) {
  return at::legacy::cuda::_th_symeig_out(e, V, self, eigenvectors, upper);
}

std::tuple<Tensor,Tensor> symeig_cuda(const Tensor & self, bool eigenvectors, bool upper) {
  return at::legacy::cuda::_th_symeig(self, eigenvectors, upper);
}

std::tuple<Tensor &,Tensor &> eig_out_cuda(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors) {
  return at::legacy::cuda::_th_eig_out(e, v, self, eigenvectors);
}

std::tuple<Tensor,Tensor> eig_cuda(const Tensor & self, bool eigenvectors) {
  return at::legacy::cuda::_th_eig(self, eigenvectors);
}

std::tuple<Tensor &,Tensor &,Tensor &> svd_out_cuda(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv) {
  return at::legacy::cuda::_th_svd_out(U, S, V, self, some, compute_uv);
}

std::tuple<Tensor,Tensor,Tensor> svd_cuda(const Tensor & self, bool some, bool compute_uv) {
  return at::legacy::cuda::_th_svd(self, some, compute_uv);
}

Tensor & cholesky_inverse_out_cuda(Tensor & result, const Tensor & self, bool upper) {
  return at::legacy::cuda::_th_potri_out(result, self, upper);
}

Tensor cholesky_inverse_cuda(const Tensor & self, bool upper) {
  return at::legacy::cuda::_th_potri(self, upper);
}

std::tuple<Tensor &,Tensor &> qr_out_cuda(Tensor & Q, Tensor & R, const Tensor & self) {
  return at::legacy::cuda::_th_qr_out(Q, R, self);
}

std::tuple<Tensor,Tensor> qr_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_qr(self);
}

std::tuple<Tensor &,Tensor &> geqrf_out_cuda(Tensor & result0, Tensor & result1, const Tensor & self) {
  return at::legacy::cuda::_th_geqrf_out(result0, result1, self);
}

std::tuple<Tensor,Tensor> geqrf_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_geqrf(self);
}

Tensor & lu_solve_out_cuda(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  return at::legacy::cuda::_th_btrisolve_out(result, self, LU_data, LU_pivots);
}

Tensor lu_solve_cuda(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
  return at::legacy::cuda::_th_btrisolve(self, LU_data, LU_pivots);
}

std::tuple<Tensor,Tensor> _multinomial_alias_setup_cuda(const Tensor & probs) {
  return at::legacy::cuda::_th_multinomial_alias_setup(probs);
}

Tensor _multinomial_alias_draw_cuda(const Tensor & q, const Tensor & J, int64_t num_samples, Generator * generator) {
  return at::legacy::cuda::_th_multinomial_alias_draw(q, J, num_samples, generator);
}

Tensor & multinomial_out_cuda(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
  return at::legacy::cuda::_th_multinomial_out(result, self, num_samples, replacement, generator);
}

Tensor multinomial_cuda(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
  return at::legacy::cuda::_th_multinomial(self, num_samples, replacement, generator);
}

Tensor & lgamma_out_cuda(Tensor & result, const Tensor & self) {
  return at::legacy::cuda::_th_lgamma_out(result, self);
}

Tensor lgamma_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_lgamma(self);
}

Tensor & digamma_out_cuda(Tensor & result, const Tensor & self) {
  return at::legacy::cuda::_th_digamma_out(result, self);
}
Tensor digamma_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_digamma(self);
}

Tensor & polygamma_out_cuda(Tensor & result, int64_t n, const Tensor & self) {
  return at::legacy::cuda::_th_polygamma_out(result, n, self);
}

Tensor polygamma_cuda(int64_t n, const Tensor & self) {
  return at::legacy::cuda::_th_polygamma(n, self);
}

Tensor & erfinv_out_cuda(Tensor & result, const Tensor & self) {
  return at::legacy::cuda::_th_erfinv_out(result, self);
}

Tensor erfinv_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_erfinv(self);
}

Tensor dist_cuda(const Tensor & self, const Tensor & other, Scalar p) {
  return at::legacy::cuda::_th_dist(self, other, p);
}

Tensor & atan2_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_atan2_out(result, self, other);
}

Tensor atan2_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_atan2(self, other);
}

Tensor & sign_out_cuda(Tensor & result, const Tensor & self) {
  return at::legacy::cuda::_th_sign_out(result, self);
}

Tensor sign_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_sign(self);
}

Tensor & fmod_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_fmod_out(result, self, other);
}

Tensor fmod_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_fmod(self, other);
}

Tensor & fmod_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_fmod_out(result, self, other);
}

Tensor fmod_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_fmod(self, other);
}

Tensor & remainder_out_cuda(Tensor & result, const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_remainder_out(result, self, other);
}

Tensor remainder_cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_remainder(self, other);
}

Tensor & remainder_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_remainder_out(result, self, other);
}

Tensor remainder_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_remainder(self, other);
}

Tensor & min_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_min_out(result, self, other);
}

Tensor min_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_min(self, other);
}

Tensor min_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_min(self);
}

Tensor & max_out_cuda(Tensor & result, const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_max_out(result, self, other);
}
Tensor max_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_max(self, other);
}

Tensor max_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_max(self);
}

std::tuple<Tensor &,Tensor &> sort_out_cuda(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
  return at::legacy::cuda::_th_sort_out(values, indices, self, dim, descending);
}

std::tuple<Tensor,Tensor> sort_cuda(const Tensor & self, int64_t dim, bool descending) {
  return at::legacy::cuda::_th_sort(self, dim, descending);
}

Tensor argsort_cuda(const Tensor & self, int64_t dim, bool descending) {
  return std::get<1>(at::legacy::cuda::_th_sort(self, dim, descending));
}

std::tuple<Tensor &,Tensor &> topk_out_cuda(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  return at::legacy::cuda::_th_topk_out(values, indices, self, k, dim, largest, sorted);
}

std::tuple<Tensor,Tensor> topk_cuda(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
  return at::legacy::cuda::_th_topk(self, k, dim, largest, sorted);
}

Tensor & renorm_out_cuda(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::legacy::cuda::_th_renorm_out(result, self, p, dim, maxnorm);
}

Tensor renorm_cuda(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
  return at::legacy::cuda::_th_renorm(self, p, dim, maxnorm);
}

Tensor unfold_cuda(const Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  return at::legacy::cuda::_th_unfold(self, dimension, size, step);
}

bool equal_cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_equal(self, other);
}

Tensor & pow_out_cuda(Tensor & result, const Tensor & self, const Tensor & exponent) {
  return at::legacy::cuda::_th_pow_out(result, self, exponent);
}

Tensor pow_cuda(const Tensor & self, const Tensor & exponent) {
  return at::legacy::cuda::_th_pow(self, exponent);
}

Tensor & pow_out_cuda(Tensor & result, Scalar self, const Tensor & exponent) {
  return at::legacy::cuda::_th_pow_out(result, self, exponent);
}

Tensor pow_cuda(Scalar self, const Tensor & exponent) {
  return at::legacy::cuda::_th_pow(self, exponent);
}

Tensor & normal_out_cuda(Tensor & output, const Tensor & mean, double std, Generator * generator) {
  return at::legacy::cuda::_th_normal_out(output, mean, std, generator);
}

Tensor normal_cuda(const Tensor & mean, double std, Generator * generator) {
  return at::legacy::cuda::_th_normal(mean, std, generator);
}

Tensor & normal_out_cuda(Tensor & output, double mean, const Tensor & std, Generator * generator) {
  return at::legacy::cuda::_th_normal_out(output, mean, std, generator);
}

Tensor normal_cuda(double mean, const Tensor & std, Generator * generator) {
  return at::legacy::cuda::_th_normal(mean, std, generator);
}

Tensor & normal_out_cuda(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) {
  return at::legacy::cuda::_th_normal_out(output, mean, std, generator);
}

Tensor normal_cuda(const Tensor & mean, const Tensor & std, Generator * generator) {
  return at::legacy::cuda::_th_normal(mean, std, generator);
}

Tensor alias_cuda(const Tensor & self) {
  return at::legacy::cuda::_th_alias(self);
}

Tensor __and___cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_and(self, other);
}

Tensor __and___cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_and(self, other);
}

Tensor __or___cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_or(self, other);
}

Tensor __or___cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_or(self, other);
}

Tensor __xor___cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_xor(self, other);
}

Tensor __xor___cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_xor(self, other);
}

Tensor __lshift___cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_lshift(self, other);
}

Tensor __lshift___cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_lshift(self, other);
}

Tensor __rshift___cuda(const Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_rshift(self, other);
}

Tensor __rshift___cuda(const Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_rshift(self, other);
}

Tensor & __iand___cuda(Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_iand_(self, other);
}

Tensor & __iand___cuda(Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_iand_(self, other);
}

Tensor & __ior___cuda(Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ior_(self, other);
}

Tensor & __ior___cuda(Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ior_(self, other);
}

Tensor & __ixor___cuda(Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ixor_(self, other);
}

Tensor & __ixor___cuda(Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ixor_(self, other);
}

Tensor & __ilshift___cuda(Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_ilshift_(self, other);
}

Tensor & __ilshift___cuda(Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_ilshift_(self, other);
}

Tensor & __irshift___cuda(Tensor & self, Scalar other) {
  return at::legacy::cuda::_th_irshift_(self, other);
}

Tensor & __irshift___cuda(Tensor & self, const Tensor & other) {
  return at::legacy::cuda::_th_irshift_(self, other);
}

Tensor _getri_single_cuda(const Tensor &self) {
  return at::legacy::cuda::_th_getri_single(self);
}

Tensor & _index_copy__cuda(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source) {
  return at::legacy::cuda::_th_index_copy_(self, dim, index, source);
}

Tensor _ger_cuda(const Tensor& self, const Tensor& vec2) {
  return at::legacy::cuda::_th_ger(self, vec2);
}

Tensor& _ger_out_cuda(Tensor& result, const Tensor& self, const Tensor& vec2) {
  return at::legacy::cuda::_th_ger_out(result, self, vec2);
}

Tensor _mm_cuda(const Tensor& self, const Tensor& mat2) {
  return at::legacy::cuda::_th_mm(self, mat2);
}

Tensor & _mm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat2) {
  return at::legacy::cuda::_th_mm_out(result, self, mat2);
}

Tensor _mv_cuda(const Tensor& self, const Tensor& vec) {
  return at::legacy::cuda::_th_mv(self, vec);
}

Tensor& _mv_out_cuda(Tensor& result, const Tensor& self, const Tensor& vec) {
  return at::legacy::cuda::_th_mv_out(result, self, vec);
}

Tensor _addmv_cuda(const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmv(self, mat, vec, beta, alpha);
}

Tensor& _addmv__cuda(Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmv_(self, mat, vec, beta, alpha);
}

Tensor& _addmv_out_cuda(Tensor &result, const Tensor& self, const Tensor& mat, const Tensor& vec, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmv_out(result, self, mat, vec, beta, alpha);
}

Tensor _addr_cuda(const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addr(self, vec1, vec2, beta, alpha);
}

Tensor& _addr__cuda(Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addr_(self, vec1, vec2, beta, alpha);
}

Tensor& _addr_out_cuda(Tensor &result, const Tensor& self, const Tensor& vec1, const Tensor& vec2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addr_out(result, self, vec1, vec2, beta, alpha);
}

Tensor _dot_cuda(const Tensor& self, const Tensor& tensor) {
  return at::legacy::cuda::_th_dot(self, tensor);
}

Tensor _cumsum_cuda(const Tensor& self, int64_t dim) {
  return at::legacy::cuda::_th_cumsum(self, dim);
}

Tensor& _cumsum_out_cuda(Tensor& result, const Tensor& self, int64_t dim) {
  return at::legacy::cuda::_th_cumsum_out(result, self, dim);
}

Tensor _cumprod_cuda(const Tensor& self, int64_t dim) {
  return at::legacy::cuda::_th_cumprod(self, dim);
}

Tensor& _cumprod_out_cuda(Tensor& result, const Tensor& self, int64_t dim) {
  return at::legacy::cuda::_th_cumprod_out(result, self, dim);
}

Tensor _var_cuda(const Tensor& self, bool unbiased) {
  return at::legacy::cuda::_th_var(self, unbiased);
}

Tensor _std_cuda(const Tensor& self, bool unbiased) {
  return at::legacy::cuda::_th_std(self, unbiased);
}

Tensor& _addmm_out_cuda(Tensor& result, const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmm_out(result, self, mat1, mat2, beta, alpha);
}

Tensor _addmm_cuda(const Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmm(self, mat1, mat2, beta, alpha);
}

Tensor& _addmm__cuda(Tensor& self, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
  return at::legacy::cuda::_th_addmm_(self, mat1, mat2, beta, alpha);
}

Tensor& fill__cuda(Tensor& self, Scalar value) {
  return at::legacy::cuda::_th_fill_(self, value);
}

Tensor& fill__cuda(Tensor& self, const Tensor& value) {
  return at::legacy::cuda::_th_fill_(self, value);
}

Tensor & _cat_out_cuda(Tensor & result, TensorList tensors, int64_t dim) {
  return at::legacy::cuda::_th_cat_out(result, tensors, dim);
}

Tensor _cat_cuda(TensorList tensors, int64_t dim) {
  return at::legacy::cuda::_th_cat(tensors, dim);
}

std::tuple<Tensor, Tensor> _mode_cuda(const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_mode(self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> _mode_out_cuda(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_mode_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> _max_cuda(const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_max(self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> _th_max_out_cuda(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_max_out(values, indices, self, dim, keepdim);
}

std::tuple<Tensor, Tensor> _min_cuda(const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_min(self, dim, keepdim);
}

std::tuple<Tensor &,Tensor &> _th_min_out_cuda(Tensor& values, Tensor& indices,
                                       const Tensor& self, int64_t dim, bool keepdim) {
  return at::legacy::cuda::_th_min_out(values, indices, self, dim, keepdim);
}

Tensor& _copy_ignoring_overlaps__cuda(Tensor& self, const Tensor& src) {
  return at::legacy::cuda::_th_copy_ignoring_overlaps_(self, src);
}

}} // namespace at::native
