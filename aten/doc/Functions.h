#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"



namespace at {

static inline Tensor & copy_out(const Tensor & src, Tensor & dst) {
  dst.resize_(src.sizes());
  dst.type().copy(src,dst);
  return dst;
}

static inline Tensor & zeros_out(IntList size, Tensor & result);
static inline Tensor & ones_out(IntList size, Tensor & result);
static inline int64_t numel(const Tensor & self);
static inline Tensor & masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result);
static inline Tensor masked_select(const Tensor & self, const Tensor & mask);
static inline Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1);
static inline Tensor t(const Tensor & self);
static inline Tensor & squeeze_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor squeeze(const Tensor & self, int64_t dim);
static inline Tensor & squeeze_out(const Tensor & self, Tensor & result);
static inline Tensor squeeze(const Tensor & self);
static inline Tensor & unsqueeze_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor unsqueeze(const Tensor & self, int64_t dim);
static inline Tensor & nonzero_out(const Tensor & self, Tensor & result);
static inline Tensor nonzero(const Tensor & self);
static inline Tensor & index_select_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result);
static inline Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor & range_out(Scalar start, Scalar end, Scalar step, Tensor & result);
static inline Tensor & range_out(Scalar start, Scalar end, Tensor & result);
static inline Tensor & gather_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result);
static inline Tensor gather(const Tensor & self, int64_t dim, const Tensor & index);
static inline bool equal(const Tensor & self, const Tensor & other);
static inline Tensor & __and___out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor __and__(const Tensor & self, Scalar value);
static inline Tensor & __and___out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor __and__(const Tensor & self, const Tensor & other);
static inline Tensor & __iand__(Tensor & self, Scalar value);
static inline Tensor & __iand__(Tensor & self, const Tensor & other);
static inline Tensor & __or___out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor __or__(const Tensor & self, Scalar value);
static inline Tensor & __or___out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor __or__(const Tensor & self, const Tensor & other);
static inline Tensor & __ior__(Tensor & self, Scalar value);
static inline Tensor & __ior__(Tensor & self, const Tensor & other);
static inline Tensor & __xor___out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor __xor__(const Tensor & self, Scalar value);
static inline Tensor & __xor___out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor __xor__(const Tensor & self, const Tensor & other);
static inline Tensor & __ixor__(Tensor & self, Scalar value);
static inline Tensor & __ixor__(Tensor & self, const Tensor & other);
static inline Tensor & __lshift___out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor __lshift__(const Tensor & self, Scalar value);
static inline Tensor & __lshift___out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor __lshift__(const Tensor & self, const Tensor & other);
static inline Tensor & __ilshift__(Tensor & self, Scalar value);
static inline Tensor & __ilshift__(Tensor & self, const Tensor & other);
static inline Tensor & __rshift___out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor __rshift__(const Tensor & self, Scalar value);
static inline Tensor & __rshift___out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor __rshift__(const Tensor & self, const Tensor & other);
static inline Tensor & __irshift__(Tensor & self, Scalar value);
static inline Tensor & __irshift__(Tensor & self, const Tensor & other);
static inline Tensor & lt_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor lt(const Tensor & tensor, Scalar value);
static inline Tensor & lt_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor lt(const Tensor & tensor, const Tensor & other);
static inline Tensor & gt_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor gt(const Tensor & tensor, Scalar value);
static inline Tensor & gt_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor gt(const Tensor & tensor, const Tensor & other);
static inline Tensor & le_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor le(const Tensor & tensor, Scalar value);
static inline Tensor & le_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor le(const Tensor & tensor, const Tensor & other);
static inline Tensor & ge_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor ge(const Tensor & tensor, Scalar value);
static inline Tensor & ge_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor ge(const Tensor & tensor, const Tensor & other);
static inline Tensor & eq_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor eq(const Tensor & tensor, Scalar value);
static inline Tensor & eq_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor eq(const Tensor & tensor, const Tensor & other);
static inline Tensor & ne_out(const Tensor & tensor, Scalar value, Tensor & result);
static inline Tensor ne(const Tensor & tensor, Scalar value);
static inline Tensor & ne_out(const Tensor & tensor, const Tensor & other, Tensor & result);
static inline Tensor ne(const Tensor & tensor, const Tensor & other);
static inline std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices);
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, Tensor & min, Tensor & min_indices);
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim);
static inline Tensor & min_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor min(const Tensor & self, const Tensor & other);
static inline Scalar min(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices);
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, Tensor & max, Tensor & max_indices);
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim);
static inline Tensor & max_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor max(const Tensor & self, const Tensor & other);
static inline Scalar max(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim);
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim);
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, bool keepdim);
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim);
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim);
static inline Scalar median(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim);
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending);
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k);
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest);
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim);
static inline Tensor & abs_out(const Tensor & self, Tensor & destination);
static inline Tensor abs(const Tensor & self);
static inline Tensor & sigmoid_out(const Tensor & self, Tensor & result);
static inline Tensor sigmoid(const Tensor & self);
static inline Tensor & log_out(const Tensor & self, Tensor & result);
static inline Tensor log(const Tensor & self);
static inline Tensor & log1p_out(const Tensor & self, Tensor & result);
static inline Tensor log1p(const Tensor & self);
static inline Tensor & lgamma_out(const Tensor & self, Tensor & result);
static inline Tensor lgamma(const Tensor & self);
static inline Tensor & exp_out(const Tensor & self, Tensor & result);
static inline Tensor exp(const Tensor & self);
static inline Tensor & cos_out(const Tensor & self, Tensor & result);
static inline Tensor cos(const Tensor & self);
static inline Tensor & acos_out(const Tensor & self, Tensor & result);
static inline Tensor acos(const Tensor & self);
static inline Tensor & cosh_out(const Tensor & self, Tensor & result);
static inline Tensor cosh(const Tensor & self);
static inline Tensor & sin_out(const Tensor & self, Tensor & result);
static inline Tensor sin(const Tensor & self);
static inline Tensor & asin_out(const Tensor & self, Tensor & result);
static inline Tensor asin(const Tensor & self);
static inline Tensor & sinh_out(const Tensor & self, Tensor & result);
static inline Tensor sinh(const Tensor & self);
static inline Tensor & tan_out(const Tensor & self, Tensor & result);
static inline Tensor tan(const Tensor & self);
static inline Tensor & atan_out(const Tensor & self, Tensor & result);
static inline Tensor atan(const Tensor & self);
static inline Tensor & tanh_out(const Tensor & self, Tensor & result);
static inline Tensor tanh(const Tensor & self);
static inline Tensor & sqrt_out(const Tensor & self, Tensor & result);
static inline Tensor sqrt(const Tensor & self);
static inline Tensor & rsqrt_out(const Tensor & self, Tensor & result);
static inline Tensor rsqrt(const Tensor & self);
static inline Tensor & ceil_out(const Tensor & self, Tensor & result);
static inline Tensor ceil(const Tensor & self);
static inline Tensor & floor_out(const Tensor & self, Tensor & result);
static inline Tensor floor(const Tensor & self);
static inline Tensor & round_out(const Tensor & self, Tensor & result);
static inline Tensor round(const Tensor & self);
static inline Tensor & trunc_out(const Tensor & self, Tensor & result);
static inline Tensor trunc(const Tensor & self);
static inline Tensor & frac_out(const Tensor & self, Tensor & result);
static inline Tensor frac(const Tensor & self);
static inline Tensor & mean_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination);
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim);
static inline Tensor & mean_out(const Tensor & self, int64_t dim, Tensor & destination);
static inline Tensor mean(const Tensor & self, int64_t dim);
static inline Scalar mean(const Tensor & self);
static inline Tensor & var_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination);
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
static inline Tensor & var_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination);
static inline Tensor var(const Tensor & self, int64_t dim, bool keepdim);
static inline Tensor & var_out(const Tensor & self, int64_t dim, Tensor & destination);
static inline Tensor var(const Tensor & self, int64_t dim);
static inline Scalar var(const Tensor & self, bool unbiased);
static inline Scalar var(const Tensor & self);
static inline Tensor & std_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination);
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
static inline Tensor & std_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination);
static inline Tensor std(const Tensor & self, int64_t dim, bool keepdim);
static inline Tensor & std_out(const Tensor & self, int64_t dim, Tensor & destination);
static inline Tensor std(const Tensor & self, int64_t dim);
static inline Scalar std(const Tensor & self, bool unbiased);
static inline Scalar std(const Tensor & self);
static inline Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor & destination);
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim);
static inline Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, Tensor & destination);
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim);
static inline Scalar norm(const Tensor & self, Scalar p);
static inline Scalar norm(const Tensor & self);
static inline Tensor & renorm_out(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor & destination);
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
static inline Scalar dist(const Tensor & self, const Tensor & other, Scalar p);
static inline Scalar dist(const Tensor & self, const Tensor & other);
static inline Tensor & reciprocal_out(const Tensor & self, Tensor & destination);
static inline Tensor reciprocal(const Tensor & self);
static inline Tensor & neg_out(const Tensor & self, Tensor & destination);
static inline Tensor neg(const Tensor & self);
static inline Tensor & atan2_out(const Tensor & self, const Tensor & other, Tensor & destination);
static inline Tensor atan2(const Tensor & self, const Tensor & other);
static inline Tensor & pow_out(const Tensor & self, Scalar exponent, Tensor & destination);
static inline Tensor pow(const Tensor & self, Scalar exponent);
static inline Tensor & pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination);
static inline Tensor pow(const Tensor & self, const Tensor & exponent);
static inline Tensor & lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination);
static inline Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight);
static inline Tensor & linspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result);
static inline Tensor & linspace_out(Scalar start, Scalar end, Tensor & result);
static inline Tensor & logspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result);
static inline Tensor & logspace_out(Scalar start, Scalar end, Tensor & result);
static inline Tensor & histc_out(const Tensor & self, Tensor & destination);
static inline Tensor histc(const Tensor & self);
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Tensor & destination);
static inline Tensor histc(const Tensor & self, int64_t bins);
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Tensor & destination);
static inline Tensor histc(const Tensor & self, int64_t bins, Scalar min);
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor & destination);
static inline Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max);
static inline Tensor & sum_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result);
static inline Tensor sum(const Tensor & self, int64_t dim, bool keepdim);
static inline Tensor & sum_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor sum(const Tensor & self, int64_t dim);
static inline Scalar sum(const Tensor & self);
static inline Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result);
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim);
static inline Tensor & prod_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor prod(const Tensor & self, int64_t dim);
static inline Scalar prod(const Tensor & self);
static inline Tensor & cumsum_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor cumsum(const Tensor & self, int64_t dim);
static inline Tensor & cumprod_out(const Tensor & self, int64_t dim, Tensor & result);
static inline Tensor cumprod(const Tensor & self, int64_t dim);
static inline Tensor & sign_out(const Tensor & self, Tensor & result);
static inline Tensor sign(const Tensor & self);
static inline Scalar trace(const Tensor & self);
static inline Tensor & add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result);
static inline Tensor add(const Tensor & self, Scalar value, const Tensor & other);
static inline Tensor & add_out(const Tensor & self, Scalar value, SparseTensor other, Tensor & result);
static inline Tensor add(const Tensor & self, Scalar value, SparseTensor other);
static inline Tensor & add_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor add(const Tensor & self, Scalar value);
static inline Tensor & add_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor add(const Tensor & self, const Tensor & other);
static inline Tensor & add_out(const Tensor & self, SparseTensor other, Tensor & result);
static inline Tensor add(const Tensor & self, SparseTensor other);
static inline Tensor & sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result);
static inline Tensor sub(const Tensor & self, Scalar value, const Tensor & other);
static inline Tensor & sub_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor sub(const Tensor & self, Scalar value);
static inline Tensor & sub_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor sub(const Tensor & self, const Tensor & other);
static inline Tensor & mul_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor mul(const Tensor & self, Scalar value);
static inline Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor mul(const Tensor & self, const Tensor & other);
static inline Tensor & div_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor div(const Tensor & self, Scalar value);
static inline Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor div(const Tensor & self, const Tensor & other);
static inline Tensor & fmod_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor fmod(const Tensor & self, Scalar value);
static inline Tensor & fmod_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor fmod(const Tensor & self, const Tensor & other);
static inline Tensor & remainder_out(const Tensor & self, Scalar value, Tensor & result);
static inline Tensor remainder(const Tensor & self, Scalar value);
static inline Tensor & remainder_out(const Tensor & self, const Tensor & other, Tensor & result);
static inline Tensor remainder(const Tensor & self, const Tensor & other);
static inline Tensor & clamp_out(const Tensor & self, Scalar min, Scalar max, Tensor & destination);
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max);
static inline Tensor & clamp_out(const Tensor & self, Scalar min, Tensor & result);
static inline Tensor clamp(const Tensor & self, Scalar min);
static inline Scalar dot(const Tensor & self, const Tensor & tensor);
static inline Tensor & tril_out(const Tensor & self, int64_t diagonal, Tensor & destination);
static inline Tensor tril(const Tensor & self, int64_t diagonal);
static inline Tensor & tril_out(const Tensor & self, Tensor & destination);
static inline Tensor tril(const Tensor & self);
static inline Tensor & triu_out(const Tensor & self, int64_t diagonal, Tensor & destination);
static inline Tensor triu(const Tensor & self, int64_t diagonal);
static inline Tensor & triu_out(const Tensor & self, Tensor & destination);
static inline Tensor triu(const Tensor & self);
static inline Tensor & cross_out(const Tensor & self, const Tensor & other, int64_t dim, Tensor & destination);
static inline Tensor cross(const Tensor & self, const Tensor & other, int64_t dim);
static inline Tensor & cross_out(const Tensor & self, const Tensor & other, Tensor & destination);
static inline Tensor cross(const Tensor & self, const Tensor & other);
static inline Tensor & eye_out(int64_t n, Tensor & result);
static inline Tensor & eye_out(int64_t n, int64_t m, Tensor & result);
static inline Tensor & diag_out(const Tensor & self, int64_t diagonal, Tensor & result);
static inline Tensor diag(const Tensor & self, int64_t diagonal);
static inline Tensor & diag_out(const Tensor & self, Tensor & result);
static inline Tensor diag(const Tensor & self);
static inline Tensor & addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result);
static inline Tensor addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2);
static inline Tensor & addmm_out(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result);
static inline Tensor addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2);
static inline Tensor & addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result);
static inline Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2);
static inline Tensor & addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result);
static inline Tensor addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec);
static inline Tensor & addmv_out(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result);
static inline Tensor addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec);
static inline Tensor & addmv_out(const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result);
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec);
static inline Tensor & addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result);
static inline Tensor addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2);
static inline Tensor & addr_out(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result);
static inline Tensor addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2);
static inline Tensor & addr_out(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result);
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2);
static inline Tensor & ger_out(const Tensor & self, const Tensor & vec2, Tensor & result);
static inline Tensor ger(const Tensor & self, const Tensor & vec2);
static inline Tensor & mv_out(const Tensor & self, const Tensor & vec, Tensor & result);
static inline Tensor mv(const Tensor & self, const Tensor & vec);
static inline Tensor & mm_out(const Tensor & self, const Tensor & mat2, Tensor & result);
static inline Tensor mm(const Tensor & self, const Tensor & mat2);
static inline Tensor & bmm_out(const Tensor & self, const Tensor & mat2, Tensor & result);
static inline Tensor bmm(const Tensor & self, const Tensor & mat2);
static inline Tensor & addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & addbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & addbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & baddbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & baddbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result);
static inline Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2);
static inline Tensor & addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result);
static inline Tensor addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2);
static inline Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result);
static inline Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2);
static inline Tensor & addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result);
static inline Tensor addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2);
static inline Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result);
static inline Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2);
static inline std::tuple<Tensor &,Tensor &> gesv_out(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu);
static inline std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor &,Tensor &> gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular);
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose);
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper);
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper);
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors);
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors);
static inline std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3);
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some);
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, Tensor & res1, Tensor & res2, Tensor & res3);
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self);
static inline Tensor & inverse_out(const Tensor & self, Tensor & output);
static inline Tensor inverse(const Tensor & self);
static inline Tensor & potrf_out(const Tensor & self, bool upper, Tensor & output);
static inline Tensor potrf(const Tensor & self, bool upper);
static inline Tensor & potrf_out(const Tensor & self, Tensor & output);
static inline Tensor potrf(const Tensor & self);
static inline Tensor & potrs_out(const Tensor & self, const Tensor & input2, bool upper, Tensor & result);
static inline Tensor potrs(const Tensor & self, const Tensor & input2, bool upper);
static inline Tensor & potrs_out(const Tensor & self, const Tensor & input2, Tensor & result);
static inline Tensor potrs(const Tensor & self, const Tensor & input2);
static inline Tensor & potri_out(const Tensor & self, bool upper, Tensor & output);
static inline Tensor potri(const Tensor & self, bool upper);
static inline Tensor & potri_out(const Tensor & self, Tensor & output);
static inline Tensor potri(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol);
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper);
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Scalar tol, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, Scalar tol);
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> qr_out(const Tensor & self, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> qr(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2);
static inline std::tuple<Tensor,Tensor> geqrf(const Tensor & self);
static inline std::tuple<Tensor &,const Tensor &> orgqr_out(const Tensor & self, const Tensor & input2, Tensor & result);
static inline std::tuple<Tensor,const Tensor &> orgqr(const Tensor & self, const Tensor & input2);
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & result);
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose);
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, Tensor & result);
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left);
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, Tensor & result);
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3);
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, bool pivot, const Tensor & self, Tensor & result, Tensor & pivots);
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & info, bool pivot, const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, const Tensor & self, Tensor & result, Tensor & pivots);
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & info, const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> btrifact_out(bool pivot, const Tensor & self, Tensor & result, Tensor & pivots);
static inline std::tuple<Tensor,Tensor> btrifact(bool pivot, const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & self, Tensor & result, Tensor & pivots);
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self);
static inline Tensor & btrisolve_out(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & result);
static inline Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots);
static inline Tensor & randperm_out(Generator & generator, int64_t n, Tensor & result);
static inline Tensor & randperm_out(int64_t n, Tensor & result);
static inline Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement, Tensor & result);
static inline Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement);
static inline Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, Tensor & result);
static inline Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples);
static inline Tensor & multinomial_out(const Tensor & self, int64_t num_samples, bool replacement, Tensor & result);
static inline Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement);
static inline Tensor & multinomial_out(const Tensor & self, int64_t num_samples, Tensor & result);
static inline Tensor multinomial(const Tensor & self, int64_t num_samples);
static inline Tensor & rand_out(Generator & generator, IntList size, Tensor & result);
static inline Tensor & rand_out(IntList size, Tensor & result);
static inline Tensor & randn_out(Generator & generator, IntList size, Tensor & result);
static inline Tensor & randn_out(IntList size, Tensor & result);
static inline Tensor & select_out(const Tensor & self, int dim, int64_t sliceIndex, Tensor & result);
static inline Tensor select(const Tensor & self, int dim, int64_t sliceIndex);
static inline Tensor & cat_out(TensorList tensors, int dim, Tensor & self);
static inline Tensor cat(TensorList tensors, int dim);
static inline void Abs_updateOutput(const Tensor & input, const Tensor & output);
static inline void Abs_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput);
static inline void AbsCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void AbsCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights);
static inline void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights);
static inline void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index);
static inline void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index);
static inline void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index);
static inline void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index);
static inline void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index);
static inline void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index);
static inline void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index);
static inline void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index);
static inline void ELU_updateOutput(const Tensor & input, const Tensor & output, Scalar alpha, bool inplace);
static inline void ELU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar alpha, bool inplace);
static inline void DistKLDivCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void DistKLDivCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void GatedLinear_updateOutput(const Tensor & input, const Tensor & output, int dim);
static inline void GatedLinear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int dim);
static inline void HardShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda);
static inline void HardShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda);
static inline void HardTanh_updateOutput(const Tensor & input, const Tensor & output, Scalar min_val, Scalar max_val, bool inplace);
static inline void HardTanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar min_val, Scalar max_val, bool inplace);
static inline void L1Cost_updateOutput(const Tensor & input, const Tensor & output);
static inline void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput);
static inline void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradInput);
static inline void LeakyReLU_updateOutput(const Tensor & input, const Tensor & output, Scalar negval, bool inplace);
static inline void LeakyReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar negval, bool inplace);
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & hx, const Tensor & output, const Tensor & storage);
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & hx, const Tensor & output, const Tensor & storage);
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & hx, const Tensor & output, const Tensor & storage);
static inline void GRUFused_updateGradInput(const Tensor & gradInInput, const Tensor & gradInHidden, const Tensor & gradOutput, const Tensor & gradInputHx, const Tensor & storage);
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & cell, const Tensor & output, const Tensor & outputCell);
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & cell, const Tensor & output, const Tensor & outputCell);
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & cell, const Tensor & output, const Tensor & outputCell);
static inline void LSTMFused_updateGradInput(const Tensor & storage, const Tensor & gradInGates, const Tensor & cx, const Tensor & cy, const Tensor & gradOutput, const Tensor & gradOutputCell, const Tensor & gradInputCx);
static inline void LogSigmoid_updateOutput(const Tensor & input, const Tensor & output, const Tensor & buffer);
static inline void LogSigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & buffer);
static inline void LogSoftMax_updateOutput(const Tensor & input, const Tensor & output);
static inline void LogSoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void MarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, Scalar margin);
static inline void MarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, Scalar margin);
static inline void SoftMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void SoftMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void MSECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void MSECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void MultiLabelMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, const Tensor & isTarget, bool sizeAverage);
static inline void MultiLabelMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, const Tensor & isTarget, bool sizeAverage);
static inline void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, const Tensor & weights, Scalar margin);
static inline void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, Scalar margin);
static inline void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, const Tensor & weights, Scalar margin);
static inline void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, Scalar margin);
static inline void PReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, int64_t nOutputPlane);
static inline void PReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int64_t nOutputPlane);
static inline void PReLU_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradWeight, const Tensor & gradWeightBuf, const Tensor & gradWeightBuf2, int64_t nOutputPlane, Scalar scale);
static inline void Linear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & addBuffer);
static inline void Linear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight);
static inline void Linear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & addBuffer, Scalar scale);
static inline void RReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace, Generator & generator);
static inline void RReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace);
static inline void Sigmoid_updateOutput(const Tensor & input, const Tensor & output);
static inline void Sigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void Sigmoid_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void SmoothL1Criterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage);
static inline void SmoothL1Criterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage);
static inline void SoftMax_updateOutput(const Tensor & input, const Tensor & output);
static inline void SoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void SoftPlus_updateOutput(const Tensor & input, const Tensor & output, Scalar beta, Scalar threshold);
static inline void SoftPlus_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar beta, Scalar threshold);
static inline void SoftShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda);
static inline void SoftShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda);
static inline void IndexLinear_updateOutput(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & normalizedValues, int train);
static inline void IndexLinear_accGradParameters(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, const Tensor & valuesBuffer, Scalar weightDecay, Scalar scale);
static inline void SparseLinear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias);
static inline void SparseLinear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, Scalar weightDecay, Scalar scale);
static inline void Sqrt_updateOutput(const Tensor & input, const Tensor & output, Scalar eps);
static inline void Sqrt_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void Square_updateOutput(const Tensor & input, const Tensor & output);
static inline void Square_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput);
static inline void Tanh_updateOutput(const Tensor & input, const Tensor & output);
static inline void Tanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void Tanh_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output);
static inline void Threshold_updateOutput(const Tensor & input, const Tensor & output, Scalar threshold, Scalar val, bool inplace);
static inline void Threshold_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar threshold, Scalar val, bool inplace);
static inline void TemporalConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize, int outputFrameSize);
static inline void TemporalConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW);
static inline void TemporalConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale);
static inline void TemporalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int dW);
static inline void TemporalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int dW);
static inline void TemporalSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize);
static inline void TemporalSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW);
static inline void TemporalSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale);
static inline void TemporalRowConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst);
static inline void TemporalRowConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst);
static inline void TemporalRowConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst, Scalar scale);
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps);
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps);
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps);
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps);
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps);
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps);
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps);
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps);
static inline void SpatialConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH);
static inline void SpatialConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH);
static inline void SpatialConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale);
static inline void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale);
static inline void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale);
static inline void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialDepthWiseConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH);
static inline void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale);
static inline void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale);
static inline void SpatialConvolutionLocal_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight);
static inline void SpatialConvolutionLocal_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight);
static inline void SpatialConvolutionLocal_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight, Scalar scale);
static inline void SpatialAdaptiveMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight);
static inline void SpatialAdaptiveMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices);
static inline void SpatialAdaptiveAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int owidth, int oheight);
static inline void SpatialAdaptiveAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput);
static inline void SpatialAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);
static inline void SpatialAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad);
static inline void SpatialFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples);
static inline void SpatialFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices);
static inline void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH);
static inline void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH);
static inline void SpatialFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH);
static inline void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale);
static inline void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale);
static inline void SpatialFullConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH);
static inline void SpatialFullConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH);
static inline void SpatialFullConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale);
static inline void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
static inline void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
static inline void SpatialDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH);
static inline void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale);
static inline void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale);
static inline void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
static inline void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
static inline void SpatialFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH);
static inline void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale);
static inline void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale);
static inline void SpatialMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);
static inline void SpatialMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode);
static inline void SpatialDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode);
static inline void SpatialDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode);
static inline void SpatialMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight);
static inline void SpatialMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int owidth, int oheight);
static inline void SpatialSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int kH, int dW, int dH);
static inline void SpatialSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int kH, int dW, int dH);
static inline void SpatialSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int kH, int dW, int dH, Scalar scale);
static inline void SpatialUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor);
static inline void SpatialUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor);
static inline void SpatialUpSamplingBilinear_updateOutput(const Tensor & input, const Tensor & output, int outputHeight, int outputWidth);
static inline void SpatialUpSamplingBilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth);
static inline void SpatialGridSamplerBilinear_updateOutput(const Tensor & input, const Tensor & grid, const Tensor & output);
static inline void SpatialGridSamplerBilinear_updateGradInput(const Tensor & input, const Tensor & gradInput, const Tensor & grid, const Tensor & gradGrid, const Tensor & gradOutput);
static inline void VolumetricAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad);
static inline void VolumetricAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad);
static inline void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale);
static inline void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale);
static inline void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale);
static inline void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale);
static inline void VolumetricFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples);
static inline void VolumetricFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices);
static inline void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH);
static inline void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH);
static inline void VolumetricFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH);
static inline void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale);
static inline void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale);
static inline void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
static inline void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
static inline void VolumetricDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH);
static inline void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale);
static inline void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale);
static inline void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
static inline void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
static inline void VolumetricFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH);
static inline void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale);
static inline void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale);
static inline void VolumetricMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode);
static inline void VolumetricMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode);
static inline void VolumetricDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode);
static inline void VolumetricDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode);
static inline void VolumetricMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void VolumetricMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH);
static inline void SpatialReflectionPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b);
static inline void SpatialReflectionPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b);
static inline void SpatialReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b);
static inline void SpatialReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b);
static inline void FeatureLPPooling_updateOutput(const Tensor & input, const Tensor & output, Scalar power, int width, int stride, bool batchMode);
static inline void FeatureLPPooling_updateGradInput(const Tensor & gradOutput, const Tensor & input, const Tensor & output, const Tensor & gradInput, Scalar power, int width, int stride, bool batchMode);
static inline void VolumetricReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback);
static inline void VolumetricReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback);
static inline void VolumetricUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor);
static inline void VolumetricUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor);
static inline void VolumetricUpSamplingTrilinear_updateOutput(const Tensor & input, const Tensor & output, int outputDepth, int outputHeight, int outputWidth);
static inline void VolumetricUpSamplingTrilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth);
static inline void SpatialCrossMapLRN_updateOutput(const Tensor & input, const Tensor & output, const Tensor & scale, int size, Scalar alpha, Scalar beta, Scalar k);
static inline void SpatialCrossMapLRN_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & scale, const Tensor & output, int size, Scalar alpha, Scalar beta, Scalar k);

static inline Type & infer_type(const Tensor & t) {
  return t.type();
}
static inline Type & infer_type(const TensorList & tl) {
  AT_ASSERT(tl.size() > 0, "expected a non-empty list of Tensors");
  return tl[0].type();
}
// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
static inline Tensor & zeros_out(IntList size, Tensor & result) {
    return infer_type(result).zeros_out(size, result);
}
static inline Tensor & ones_out(IntList size, Tensor & result) {
    return infer_type(result).ones_out(size, result);
}
static inline int64_t numel(const Tensor & self) {
    return infer_type(self).numel(self);
}
static inline Tensor & masked_select_out(const Tensor & self, const Tensor & mask, Tensor & result) {
    return infer_type(self).masked_select_out(self, mask, result);
}
static inline Tensor masked_select(const Tensor & self, const Tensor & mask) {
    return infer_type(self).masked_select(self, mask);
}
static inline Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1) {
    return infer_type(self).transpose(self, dim0, dim1);
}
static inline Tensor t(const Tensor & self) {
    return infer_type(self).t(self);
}
static inline Tensor & squeeze_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).squeeze_out(self, dim, result);
}
static inline Tensor squeeze(const Tensor & self, int64_t dim) {
    return infer_type(self).squeeze(self, dim);
}
static inline Tensor & squeeze_out(const Tensor & self, Tensor & result) {
    return infer_type(self).squeeze_out(self, result);
}
static inline Tensor squeeze(const Tensor & self) {
    return infer_type(self).squeeze(self);
}
static inline Tensor & unsqueeze_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).unsqueeze_out(self, dim, result);
}
static inline Tensor unsqueeze(const Tensor & self, int64_t dim) {
    return infer_type(self).unsqueeze(self, dim);
}
static inline Tensor & nonzero_out(const Tensor & self, Tensor & result) {
    return infer_type(self).nonzero_out(self, result);
}
static inline Tensor nonzero(const Tensor & self) {
    return infer_type(self).nonzero(self);
}
static inline Tensor & index_select_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
    return infer_type(self).index_select_out(self, dim, index, result);
}
static inline Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).index_select(self, dim, index);
}
static inline Tensor & range_out(Scalar start, Scalar end, Scalar step, Tensor & result) {
    return infer_type(result).range_out(start, end, step, result);
}
static inline Tensor & range_out(Scalar start, Scalar end, Tensor & result) {
    return infer_type(result).range_out(start, end, result);
}
static inline Tensor & gather_out(const Tensor & self, int64_t dim, const Tensor & index, Tensor & result) {
    return infer_type(self).gather_out(self, dim, index, result);
}
static inline Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).gather(self, dim, index);
}
static inline bool equal(const Tensor & self, const Tensor & other) {
    return infer_type(self).equal(self, other);
}
static inline Tensor & __and___out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).__and___out(self, value, result);
}
static inline Tensor __and__(const Tensor & self, Scalar value) {
    return infer_type(self).__and__(self, value);
}
static inline Tensor & __and___out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).__and___out(self, other, result);
}
static inline Tensor __and__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__and__(self, other);
}
static inline Tensor & __iand__(Tensor & self, Scalar value) {
    return infer_type(self).__iand__(self, value);
}
static inline Tensor & __iand__(Tensor & self, const Tensor & other) {
    return infer_type(self).__iand__(self, other);
}
static inline Tensor & __or___out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).__or___out(self, value, result);
}
static inline Tensor __or__(const Tensor & self, Scalar value) {
    return infer_type(self).__or__(self, value);
}
static inline Tensor & __or___out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).__or___out(self, other, result);
}
static inline Tensor __or__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__or__(self, other);
}
static inline Tensor & __ior__(Tensor & self, Scalar value) {
    return infer_type(self).__ior__(self, value);
}
static inline Tensor & __ior__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ior__(self, other);
}
static inline Tensor & __xor___out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).__xor___out(self, value, result);
}
static inline Tensor __xor__(const Tensor & self, Scalar value) {
    return infer_type(self).__xor__(self, value);
}
static inline Tensor & __xor___out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).__xor___out(self, other, result);
}
static inline Tensor __xor__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__xor__(self, other);
}
static inline Tensor & __ixor__(Tensor & self, Scalar value) {
    return infer_type(self).__ixor__(self, value);
}
static inline Tensor & __ixor__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ixor__(self, other);
}
static inline Tensor & __lshift___out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).__lshift___out(self, value, result);
}
static inline Tensor __lshift__(const Tensor & self, Scalar value) {
    return infer_type(self).__lshift__(self, value);
}
static inline Tensor & __lshift___out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).__lshift___out(self, other, result);
}
static inline Tensor __lshift__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__lshift__(self, other);
}
static inline Tensor & __ilshift__(Tensor & self, Scalar value) {
    return infer_type(self).__ilshift__(self, value);
}
static inline Tensor & __ilshift__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ilshift__(self, other);
}
static inline Tensor & __rshift___out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).__rshift___out(self, value, result);
}
static inline Tensor __rshift__(const Tensor & self, Scalar value) {
    return infer_type(self).__rshift__(self, value);
}
static inline Tensor & __rshift___out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).__rshift___out(self, other, result);
}
static inline Tensor __rshift__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__rshift__(self, other);
}
static inline Tensor & __irshift__(Tensor & self, Scalar value) {
    return infer_type(self).__irshift__(self, value);
}
static inline Tensor & __irshift__(Tensor & self, const Tensor & other) {
    return infer_type(self).__irshift__(self, other);
}
static inline Tensor & lt_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).lt_out(tensor, value, result);
}
static inline Tensor lt(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).lt(tensor, value);
}
static inline Tensor & lt_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).lt_out(tensor, other, result);
}
static inline Tensor lt(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).lt(tensor, other);
}
static inline Tensor & gt_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).gt_out(tensor, value, result);
}
static inline Tensor gt(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).gt(tensor, value);
}
static inline Tensor & gt_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).gt_out(tensor, other, result);
}
static inline Tensor gt(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).gt(tensor, other);
}
static inline Tensor & le_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).le_out(tensor, value, result);
}
static inline Tensor le(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).le(tensor, value);
}
static inline Tensor & le_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).le_out(tensor, other, result);
}
static inline Tensor le(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).le(tensor, other);
}
static inline Tensor & ge_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).ge_out(tensor, value, result);
}
static inline Tensor ge(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).ge(tensor, value);
}
static inline Tensor & ge_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).ge_out(tensor, other, result);
}
static inline Tensor ge(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).ge(tensor, other);
}
static inline Tensor & eq_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).eq_out(tensor, value, result);
}
static inline Tensor eq(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).eq(tensor, value);
}
static inline Tensor & eq_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).eq_out(tensor, other, result);
}
static inline Tensor eq(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).eq(tensor, other);
}
static inline Tensor & ne_out(const Tensor & tensor, Scalar value, Tensor & result) {
    return infer_type(tensor).ne_out(tensor, value, result);
}
static inline Tensor ne(const Tensor & tensor, Scalar value) {
    return infer_type(tensor).ne(tensor, value);
}
static inline Tensor & ne_out(const Tensor & tensor, const Tensor & other, Tensor & result) {
    return infer_type(tensor).ne_out(tensor, other, result);
}
static inline Tensor ne(const Tensor & tensor, const Tensor & other) {
    return infer_type(tensor).ne(tensor, other);
}
static inline std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & min, Tensor & min_indices) {
    return infer_type(self).min_out(self, dim, keepdim, min, min_indices);
}
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).min(self, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> min_out(const Tensor & self, int64_t dim, Tensor & min, Tensor & min_indices) {
    return infer_type(self).min_out(self, dim, min, min_indices);
}
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim) {
    return infer_type(self).min(self, dim);
}
static inline Tensor & min_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).min_out(self, other, result);
}
static inline Tensor min(const Tensor & self, const Tensor & other) {
    return infer_type(self).min(self, other);
}
static inline Scalar min(const Tensor & self) {
    return infer_type(self).min(self);
}
static inline std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & max, Tensor & max_indices) {
    return infer_type(self).max_out(self, dim, keepdim, max, max_indices);
}
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).max(self, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> max_out(const Tensor & self, int64_t dim, Tensor & max, Tensor & max_indices) {
    return infer_type(self).max_out(self, dim, max, max_indices);
}
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim) {
    return infer_type(self).max(self, dim);
}
static inline Tensor & max_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).max_out(self, other, result);
}
static inline Tensor max(const Tensor & self, const Tensor & other) {
    return infer_type(self).max(self, other);
}
static inline Scalar max(const Tensor & self) {
    return infer_type(self).max(self);
}
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).kthvalue_out(self, k, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, bool keepdim) {
    return infer_type(self).kthvalue(self, k, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) {
    return infer_type(self).kthvalue_out(self, k, values, indices);
}
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k) {
    return infer_type(self).kthvalue(self, k);
}
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).kthvalue_out(self, k, dim, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    return infer_type(self).kthvalue(self, k, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) {
    return infer_type(self).kthvalue_out(self, k, dim, values, indices);
}
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim) {
    return infer_type(self).kthvalue(self, k, dim);
}
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).mode_out(self, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, bool keepdim) {
    return infer_type(self).mode(self, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, Tensor & values, Tensor & indices) {
    return infer_type(self).mode_out(self, values, indices);
}
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self) {
    return infer_type(self).mode(self);
}
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).mode_out(self, dim, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mode(self, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> mode_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    return infer_type(self).mode_out(self, dim, values, indices);
}
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim) {
    return infer_type(self).mode(self, dim);
}
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).median_out(self, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, bool keepdim) {
    return infer_type(self).median(self, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    return infer_type(self).median_out(self, dim, values, indices);
}
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim) {
    return infer_type(self).median(self, dim);
}
static inline std::tuple<Tensor &,Tensor &> median_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & values, Tensor & indices) {
    return infer_type(self).median_out(self, dim, keepdim, values, indices);
}
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).median(self, dim, keepdim);
}
static inline Scalar median(const Tensor & self) {
    return infer_type(self).median(self);
}
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, Tensor & values, Tensor & indices) {
    return infer_type(self).sort_out(self, values, indices);
}
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self) {
    return infer_type(self).sort(self);
}
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, Tensor & values, Tensor & indices) {
    return infer_type(self).sort_out(self, dim, values, indices);
}
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim) {
    return infer_type(self).sort(self, dim);
}
static inline std::tuple<Tensor &,Tensor &> sort_out(const Tensor & self, int64_t dim, bool descending, Tensor & values, Tensor & indices) {
    return infer_type(self).sort_out(self, dim, descending, values, indices);
}
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
    return infer_type(self).sort(self, dim, descending);
}
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, Tensor & values, Tensor & indices) {
    return infer_type(self).topk_out(self, k, values, indices);
}
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k) {
    return infer_type(self).topk(self, k);
}
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, Tensor & values, Tensor & indices) {
    return infer_type(self).topk_out(self, k, dim, largest, sorted, values, indices);
}
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    return infer_type(self).topk(self, k, dim, largest, sorted);
}
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, bool largest, Tensor & values, Tensor & indices) {
    return infer_type(self).topk_out(self, k, dim, largest, values, indices);
}
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest) {
    return infer_type(self).topk(self, k, dim, largest);
}
static inline std::tuple<Tensor &,Tensor &> topk_out(const Tensor & self, int64_t k, int64_t dim, Tensor & values, Tensor & indices) {
    return infer_type(self).topk_out(self, k, dim, values, indices);
}
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim) {
    return infer_type(self).topk(self, k, dim);
}
static inline Tensor & abs_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).abs_out(self, destination);
}
static inline Tensor abs(const Tensor & self) {
    return infer_type(self).abs(self);
}
static inline Tensor & sigmoid_out(const Tensor & self, Tensor & result) {
    return infer_type(self).sigmoid_out(self, result);
}
static inline Tensor sigmoid(const Tensor & self) {
    return infer_type(self).sigmoid(self);
}
static inline Tensor & log_out(const Tensor & self, Tensor & result) {
    return infer_type(self).log_out(self, result);
}
static inline Tensor log(const Tensor & self) {
    return infer_type(self).log(self);
}
static inline Tensor & log1p_out(const Tensor & self, Tensor & result) {
    return infer_type(self).log1p_out(self, result);
}
static inline Tensor log1p(const Tensor & self) {
    return infer_type(self).log1p(self);
}
static inline Tensor & lgamma_out(const Tensor & self, Tensor & result) {
    return infer_type(self).lgamma_out(self, result);
}
static inline Tensor lgamma(const Tensor & self) {
    return infer_type(self).lgamma(self);
}
static inline Tensor & exp_out(const Tensor & self, Tensor & result) {
    return infer_type(self).exp_out(self, result);
}
static inline Tensor exp(const Tensor & self) {
    return infer_type(self).exp(self);
}
static inline Tensor & cos_out(const Tensor & self, Tensor & result) {
    return infer_type(self).cos_out(self, result);
}
static inline Tensor cos(const Tensor & self) {
    return infer_type(self).cos(self);
}
static inline Tensor & acos_out(const Tensor & self, Tensor & result) {
    return infer_type(self).acos_out(self, result);
}
static inline Tensor acos(const Tensor & self) {
    return infer_type(self).acos(self);
}
static inline Tensor & cosh_out(const Tensor & self, Tensor & result) {
    return infer_type(self).cosh_out(self, result);
}
static inline Tensor cosh(const Tensor & self) {
    return infer_type(self).cosh(self);
}
static inline Tensor & sin_out(const Tensor & self, Tensor & result) {
    return infer_type(self).sin_out(self, result);
}
static inline Tensor sin(const Tensor & self) {
    return infer_type(self).sin(self);
}
static inline Tensor & asin_out(const Tensor & self, Tensor & result) {
    return infer_type(self).asin_out(self, result);
}
static inline Tensor asin(const Tensor & self) {
    return infer_type(self).asin(self);
}
static inline Tensor & sinh_out(const Tensor & self, Tensor & result) {
    return infer_type(self).sinh_out(self, result);
}
static inline Tensor sinh(const Tensor & self) {
    return infer_type(self).sinh(self);
}
static inline Tensor & tan_out(const Tensor & self, Tensor & result) {
    return infer_type(self).tan_out(self, result);
}
static inline Tensor tan(const Tensor & self) {
    return infer_type(self).tan(self);
}
static inline Tensor & atan_out(const Tensor & self, Tensor & result) {
    return infer_type(self).atan_out(self, result);
}
static inline Tensor atan(const Tensor & self) {
    return infer_type(self).atan(self);
}
static inline Tensor & tanh_out(const Tensor & self, Tensor & result) {
    return infer_type(self).tanh_out(self, result);
}
static inline Tensor tanh(const Tensor & self) {
    return infer_type(self).tanh(self);
}
static inline Tensor & sqrt_out(const Tensor & self, Tensor & result) {
    return infer_type(self).sqrt_out(self, result);
}
static inline Tensor sqrt(const Tensor & self) {
    return infer_type(self).sqrt(self);
}
static inline Tensor & rsqrt_out(const Tensor & self, Tensor & result) {
    return infer_type(self).rsqrt_out(self, result);
}
static inline Tensor rsqrt(const Tensor & self) {
    return infer_type(self).rsqrt(self);
}
static inline Tensor & ceil_out(const Tensor & self, Tensor & result) {
    return infer_type(self).ceil_out(self, result);
}
static inline Tensor ceil(const Tensor & self) {
    return infer_type(self).ceil(self);
}
static inline Tensor & floor_out(const Tensor & self, Tensor & result) {
    return infer_type(self).floor_out(self, result);
}
static inline Tensor floor(const Tensor & self) {
    return infer_type(self).floor(self);
}
static inline Tensor & round_out(const Tensor & self, Tensor & result) {
    return infer_type(self).round_out(self, result);
}
static inline Tensor round(const Tensor & self) {
    return infer_type(self).round(self);
}
static inline Tensor & trunc_out(const Tensor & self, Tensor & result) {
    return infer_type(self).trunc_out(self, result);
}
static inline Tensor trunc(const Tensor & self) {
    return infer_type(self).trunc(self);
}
static inline Tensor & frac_out(const Tensor & self, Tensor & result) {
    return infer_type(self).frac_out(self, result);
}
static inline Tensor frac(const Tensor & self) {
    return infer_type(self).frac(self);
}
static inline Tensor & mean_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    return infer_type(self).mean_out(self, dim, keepdim, destination);
}
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mean(self, dim, keepdim);
}
static inline Tensor & mean_out(const Tensor & self, int64_t dim, Tensor & destination) {
    return infer_type(self).mean_out(self, dim, destination);
}
static inline Tensor mean(const Tensor & self, int64_t dim) {
    return infer_type(self).mean(self, dim);
}
static inline Scalar mean(const Tensor & self) {
    return infer_type(self).mean(self);
}
static inline Tensor & var_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) {
    return infer_type(self).var_out(self, dim, unbiased, keepdim, destination);
}
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).var(self, dim, unbiased, keepdim);
}
static inline Tensor & var_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    return infer_type(self).var_out(self, dim, keepdim, destination);
}
static inline Tensor var(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).var(self, dim, keepdim);
}
static inline Tensor & var_out(const Tensor & self, int64_t dim, Tensor & destination) {
    return infer_type(self).var_out(self, dim, destination);
}
static inline Tensor var(const Tensor & self, int64_t dim) {
    return infer_type(self).var(self, dim);
}
static inline Scalar var(const Tensor & self, bool unbiased) {
    return infer_type(self).var(self, unbiased);
}
static inline Scalar var(const Tensor & self) {
    return infer_type(self).var(self);
}
static inline Tensor & std_out(const Tensor & self, int64_t dim, bool unbiased, bool keepdim, Tensor & destination) {
    return infer_type(self).std_out(self, dim, unbiased, keepdim, destination);
}
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).std(self, dim, unbiased, keepdim);
}
static inline Tensor & std_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & destination) {
    return infer_type(self).std_out(self, dim, keepdim, destination);
}
static inline Tensor std(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).std(self, dim, keepdim);
}
static inline Tensor & std_out(const Tensor & self, int64_t dim, Tensor & destination) {
    return infer_type(self).std_out(self, dim, destination);
}
static inline Tensor std(const Tensor & self, int64_t dim) {
    return infer_type(self).std(self, dim);
}
static inline Scalar std(const Tensor & self, bool unbiased) {
    return infer_type(self).std(self, unbiased);
}
static inline Scalar std(const Tensor & self) {
    return infer_type(self).std(self);
}
static inline Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, bool keepdim, Tensor & destination) {
    return infer_type(self).norm_out(self, p, dim, keepdim, destination);
}
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    return infer_type(self).norm(self, p, dim, keepdim);
}
static inline Tensor & norm_out(const Tensor & self, Scalar p, int64_t dim, Tensor & destination) {
    return infer_type(self).norm_out(self, p, dim, destination);
}
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim) {
    return infer_type(self).norm(self, p, dim);
}
static inline Scalar norm(const Tensor & self, Scalar p) {
    return infer_type(self).norm(self, p);
}
static inline Scalar norm(const Tensor & self) {
    return infer_type(self).norm(self);
}
static inline Tensor & renorm_out(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm, Tensor & destination) {
    return infer_type(self).renorm_out(self, p, dim, maxnorm, destination);
}
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    return infer_type(self).renorm(self, p, dim, maxnorm);
}
static inline Scalar dist(const Tensor & self, const Tensor & other, Scalar p) {
    return infer_type(self).dist(self, other, p);
}
static inline Scalar dist(const Tensor & self, const Tensor & other) {
    return infer_type(self).dist(self, other);
}
static inline Tensor & reciprocal_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).reciprocal_out(self, destination);
}
static inline Tensor reciprocal(const Tensor & self) {
    return infer_type(self).reciprocal(self);
}
static inline Tensor & neg_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).neg_out(self, destination);
}
static inline Tensor neg(const Tensor & self) {
    return infer_type(self).neg(self);
}
static inline Tensor & atan2_out(const Tensor & self, const Tensor & other, Tensor & destination) {
    return infer_type(self).atan2_out(self, other, destination);
}
static inline Tensor atan2(const Tensor & self, const Tensor & other) {
    return infer_type(self).atan2(self, other);
}
static inline Tensor & pow_out(const Tensor & self, Scalar exponent, Tensor & destination) {
    return infer_type(self).pow_out(self, exponent, destination);
}
static inline Tensor pow(const Tensor & self, Scalar exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & pow_out(const Tensor & self, const Tensor & exponent, Tensor & destination) {
    return infer_type(self).pow_out(self, exponent, destination);
}
static inline Tensor pow(const Tensor & self, const Tensor & exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & lerp_out(const Tensor & self, const Tensor & end, Scalar weight, Tensor & destination) {
    return infer_type(self).lerp_out(self, end, weight, destination);
}
static inline Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) {
    return infer_type(self).lerp(self, end, weight);
}
static inline Tensor & linspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) {
    return infer_type(result).linspace_out(start, end, steps, result);
}
static inline Tensor & linspace_out(Scalar start, Scalar end, Tensor & result) {
    return infer_type(result).linspace_out(start, end, result);
}
static inline Tensor & logspace_out(Scalar start, Scalar end, int64_t steps, Tensor & result) {
    return infer_type(result).logspace_out(start, end, steps, result);
}
static inline Tensor & logspace_out(Scalar start, Scalar end, Tensor & result) {
    return infer_type(result).logspace_out(start, end, result);
}
static inline Tensor & histc_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).histc_out(self, destination);
}
static inline Tensor histc(const Tensor & self) {
    return infer_type(self).histc(self);
}
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Tensor & destination) {
    return infer_type(self).histc_out(self, bins, destination);
}
static inline Tensor histc(const Tensor & self, int64_t bins) {
    return infer_type(self).histc(self, bins);
}
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Tensor & destination) {
    return infer_type(self).histc_out(self, bins, min, destination);
}
static inline Tensor histc(const Tensor & self, int64_t bins, Scalar min) {
    return infer_type(self).histc(self, bins, min);
}
static inline Tensor & histc_out(const Tensor & self, int64_t bins, Scalar min, Scalar max, Tensor & destination) {
    return infer_type(self).histc_out(self, bins, min, max, destination);
}
static inline Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    return infer_type(self).histc(self, bins, min, max);
}
static inline Tensor & sum_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) {
    return infer_type(self).sum_out(self, dim, keepdim, result);
}
static inline Tensor sum(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).sum(self, dim, keepdim);
}
static inline Tensor & sum_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).sum_out(self, dim, result);
}
static inline Tensor sum(const Tensor & self, int64_t dim) {
    return infer_type(self).sum(self, dim);
}
static inline Scalar sum(const Tensor & self) {
    return infer_type(self).sum(self);
}
static inline Tensor & prod_out(const Tensor & self, int64_t dim, bool keepdim, Tensor & result) {
    return infer_type(self).prod_out(self, dim, keepdim, result);
}
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).prod(self, dim, keepdim);
}
static inline Tensor & prod_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).prod_out(self, dim, result);
}
static inline Tensor prod(const Tensor & self, int64_t dim) {
    return infer_type(self).prod(self, dim);
}
static inline Scalar prod(const Tensor & self) {
    return infer_type(self).prod(self);
}
static inline Tensor & cumsum_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).cumsum_out(self, dim, result);
}
static inline Tensor cumsum(const Tensor & self, int64_t dim) {
    return infer_type(self).cumsum(self, dim);
}
static inline Tensor & cumprod_out(const Tensor & self, int64_t dim, Tensor & result) {
    return infer_type(self).cumprod_out(self, dim, result);
}
static inline Tensor cumprod(const Tensor & self, int64_t dim) {
    return infer_type(self).cumprod(self, dim);
}
static inline Tensor & sign_out(const Tensor & self, Tensor & result) {
    return infer_type(self).sign_out(self, result);
}
static inline Tensor sign(const Tensor & self) {
    return infer_type(self).sign(self);
}
static inline Scalar trace(const Tensor & self) {
    return infer_type(self).trace(self);
}
static inline Tensor & add_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) {
    return infer_type(self).add_out(self, value, other, result);
}
static inline Tensor add(const Tensor & self, Scalar value, const Tensor & other) {
    return infer_type(self).add(self, value, other);
}
static inline Tensor & add_out(const Tensor & self, Scalar value, SparseTensor other, Tensor & result) {
    return infer_type(self).add_out(self, value, other, result);
}
static inline Tensor add(const Tensor & self, Scalar value, SparseTensor other) {
    return infer_type(self).add(self, value, other);
}
static inline Tensor & add_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).add_out(self, value, result);
}
static inline Tensor add(const Tensor & self, Scalar value) {
    return infer_type(self).add(self, value);
}
static inline Tensor & add_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).add_out(self, other, result);
}
static inline Tensor add(const Tensor & self, const Tensor & other) {
    return infer_type(self).add(self, other);
}
static inline Tensor & add_out(const Tensor & self, SparseTensor other, Tensor & result) {
    return infer_type(self).add_out(self, other, result);
}
static inline Tensor add(const Tensor & self, SparseTensor other) {
    return infer_type(self).add(self, other);
}
static inline Tensor & sub_out(const Tensor & self, Scalar value, const Tensor & other, Tensor & result) {
    return infer_type(self).sub_out(self, value, other, result);
}
static inline Tensor sub(const Tensor & self, Scalar value, const Tensor & other) {
    return infer_type(self).sub(self, value, other);
}
static inline Tensor & sub_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).sub_out(self, value, result);
}
static inline Tensor sub(const Tensor & self, Scalar value) {
    return infer_type(self).sub(self, value);
}
static inline Tensor & sub_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).sub_out(self, other, result);
}
static inline Tensor sub(const Tensor & self, const Tensor & other) {
    return infer_type(self).sub(self, other);
}
static inline Tensor & mul_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).mul_out(self, value, result);
}
static inline Tensor mul(const Tensor & self, Scalar value) {
    return infer_type(self).mul(self, value);
}
static inline Tensor & mul_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).mul_out(self, other, result);
}
static inline Tensor mul(const Tensor & self, const Tensor & other) {
    return infer_type(self).mul(self, other);
}
static inline Tensor & div_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).div_out(self, value, result);
}
static inline Tensor div(const Tensor & self, Scalar value) {
    return infer_type(self).div(self, value);
}
static inline Tensor & div_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).div_out(self, other, result);
}
static inline Tensor div(const Tensor & self, const Tensor & other) {
    return infer_type(self).div(self, other);
}
static inline Tensor & fmod_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).fmod_out(self, value, result);
}
static inline Tensor fmod(const Tensor & self, Scalar value) {
    return infer_type(self).fmod(self, value);
}
static inline Tensor & fmod_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).fmod_out(self, other, result);
}
static inline Tensor fmod(const Tensor & self, const Tensor & other) {
    return infer_type(self).fmod(self, other);
}
static inline Tensor & remainder_out(const Tensor & self, Scalar value, Tensor & result) {
    return infer_type(self).remainder_out(self, value, result);
}
static inline Tensor remainder(const Tensor & self, Scalar value) {
    return infer_type(self).remainder(self, value);
}
static inline Tensor & remainder_out(const Tensor & self, const Tensor & other, Tensor & result) {
    return infer_type(self).remainder_out(self, other, result);
}
static inline Tensor remainder(const Tensor & self, const Tensor & other) {
    return infer_type(self).remainder(self, other);
}
static inline Tensor & clamp_out(const Tensor & self, Scalar min, Scalar max, Tensor & destination) {
    return infer_type(self).clamp_out(self, min, max, destination);
}
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp(self, min, max);
}
static inline Tensor & clamp_out(const Tensor & self, Scalar min, Tensor & result) {
    return infer_type(self).clamp_out(self, min, result);
}
static inline Tensor clamp(const Tensor & self, Scalar min) {
    return infer_type(self).clamp(self, min);
}
static inline Scalar dot(const Tensor & self, const Tensor & tensor) {
    return infer_type(self).dot(self, tensor);
}
static inline Tensor & tril_out(const Tensor & self, int64_t diagonal, Tensor & destination) {
    return infer_type(self).tril_out(self, diagonal, destination);
}
static inline Tensor tril(const Tensor & self, int64_t diagonal) {
    return infer_type(self).tril(self, diagonal);
}
static inline Tensor & tril_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).tril_out(self, destination);
}
static inline Tensor tril(const Tensor & self) {
    return infer_type(self).tril(self);
}
static inline Tensor & triu_out(const Tensor & self, int64_t diagonal, Tensor & destination) {
    return infer_type(self).triu_out(self, diagonal, destination);
}
static inline Tensor triu(const Tensor & self, int64_t diagonal) {
    return infer_type(self).triu(self, diagonal);
}
static inline Tensor & triu_out(const Tensor & self, Tensor & destination) {
    return infer_type(self).triu_out(self, destination);
}
static inline Tensor triu(const Tensor & self) {
    return infer_type(self).triu(self);
}
static inline Tensor & cross_out(const Tensor & self, const Tensor & other, int64_t dim, Tensor & destination) {
    return infer_type(self).cross_out(self, other, dim, destination);
}
static inline Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) {
    return infer_type(self).cross(self, other, dim);
}
static inline Tensor & cross_out(const Tensor & self, const Tensor & other, Tensor & destination) {
    return infer_type(self).cross_out(self, other, destination);
}
static inline Tensor cross(const Tensor & self, const Tensor & other) {
    return infer_type(self).cross(self, other);
}
static inline Tensor & eye_out(int64_t n, Tensor & result) {
    return infer_type(result).eye_out(n, result);
}
static inline Tensor & eye_out(int64_t n, int64_t m, Tensor & result) {
    return infer_type(result).eye_out(n, m, result);
}
static inline Tensor & diag_out(const Tensor & self, int64_t diagonal, Tensor & result) {
    return infer_type(self).diag_out(self, diagonal, result);
}
static inline Tensor diag(const Tensor & self, int64_t diagonal) {
    return infer_type(self).diag(self, diagonal);
}
static inline Tensor & diag_out(const Tensor & self, Tensor & result) {
    return infer_type(self).diag_out(self, result);
}
static inline Tensor diag(const Tensor & self) {
    return infer_type(self).diag(self);
}
static inline Tensor & addmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    return infer_type(self).addmm_out(beta, self, alpha, mat1, mat2, result);
}
static inline Tensor addmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat1, const Tensor & mat2) {
    return infer_type(self).addmm(beta, self, alpha, mat1, mat2);
}
static inline Tensor & addmm_out(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    return infer_type(self).addmm_out(beta, self, mat1, mat2, result);
}
static inline Tensor addmm(Scalar beta, const Tensor & self, const Tensor & mat1, const Tensor & mat2) {
    return infer_type(self).addmm(beta, self, mat1, mat2);
}
static inline Tensor & addmm_out(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Tensor & result) {
    return infer_type(self).addmm_out(self, mat1, mat2, result);
}
static inline Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2) {
    return infer_type(self).addmm(self, mat1, mat2);
}
static inline Tensor & addmv_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec, Tensor & result) {
    return infer_type(self).addmv_out(beta, self, alpha, mat, vec, result);
}
static inline Tensor addmv(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & mat, const Tensor & vec) {
    return infer_type(self).addmv(beta, self, alpha, mat, vec);
}
static inline Tensor & addmv_out(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) {
    return infer_type(self).addmv_out(beta, self, mat, vec, result);
}
static inline Tensor addmv(Scalar beta, const Tensor & self, const Tensor & mat, const Tensor & vec) {
    return infer_type(self).addmv(beta, self, mat, vec);
}
static inline Tensor & addmv_out(const Tensor & self, const Tensor & mat, const Tensor & vec, Tensor & result) {
    return infer_type(self).addmv_out(self, mat, vec, result);
}
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec) {
    return infer_type(self).addmv(self, mat, vec);
}
static inline Tensor & addr_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    return infer_type(self).addr_out(beta, self, alpha, vec1, vec2, result);
}
static inline Tensor addr(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & vec1, const Tensor & vec2) {
    return infer_type(self).addr(beta, self, alpha, vec1, vec2);
}
static inline Tensor & addr_out(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    return infer_type(self).addr_out(beta, self, vec1, vec2, result);
}
static inline Tensor addr(Scalar beta, const Tensor & self, const Tensor & vec1, const Tensor & vec2) {
    return infer_type(self).addr(beta, self, vec1, vec2);
}
static inline Tensor & addr_out(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Tensor & result) {
    return infer_type(self).addr_out(self, vec1, vec2, result);
}
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2) {
    return infer_type(self).addr(self, vec1, vec2);
}
static inline Tensor & ger_out(const Tensor & self, const Tensor & vec2, Tensor & result) {
    return infer_type(self).ger_out(self, vec2, result);
}
static inline Tensor ger(const Tensor & self, const Tensor & vec2) {
    return infer_type(self).ger(self, vec2);
}
static inline Tensor & mv_out(const Tensor & self, const Tensor & vec, Tensor & result) {
    return infer_type(self).mv_out(self, vec, result);
}
static inline Tensor mv(const Tensor & self, const Tensor & vec) {
    return infer_type(self).mv(self, vec);
}
static inline Tensor & mm_out(const Tensor & self, const Tensor & mat2, Tensor & result) {
    return infer_type(self).mm_out(self, mat2, result);
}
static inline Tensor mm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self).mm(self, mat2);
}
static inline Tensor & bmm_out(const Tensor & self, const Tensor & mat2, Tensor & result) {
    return infer_type(self).bmm_out(self, mat2, result);
}
static inline Tensor bmm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self).bmm(self, mat2);
}
static inline Tensor & addbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).addbmm_out(beta, self, alpha, batch1, batch2, result);
}
static inline Tensor addbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).addbmm(beta, self, alpha, batch1, batch2);
}
static inline Tensor & addbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).addbmm_out(beta, self, batch1, batch2, result);
}
static inline Tensor addbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).addbmm(beta, self, batch1, batch2);
}
static inline Tensor & addbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).addbmm_out(self, batch1, batch2, result);
}
static inline Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).addbmm(self, batch1, batch2);
}
static inline Tensor & baddbmm_out(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).baddbmm_out(beta, self, alpha, batch1, batch2, result);
}
static inline Tensor baddbmm(Scalar beta, const Tensor & self, Scalar alpha, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).baddbmm(beta, self, alpha, batch1, batch2);
}
static inline Tensor & baddbmm_out(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).baddbmm_out(beta, self, batch1, batch2, result);
}
static inline Tensor baddbmm(Scalar beta, const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).baddbmm(beta, self, batch1, batch2);
}
static inline Tensor & baddbmm_out(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Tensor & result) {
    return infer_type(self).baddbmm_out(self, batch1, batch2, result);
}
static inline Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2) {
    return infer_type(self).baddbmm(self, batch1, batch2);
}
static inline Tensor & addcmul_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    return infer_type(self).addcmul_out(self, value, tensor1, tensor2, result);
}
static inline Tensor addcmul(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    return infer_type(self).addcmul(self, value, tensor1, tensor2);
}
static inline Tensor & addcmul_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    return infer_type(self).addcmul_out(self, tensor1, tensor2, result);
}
static inline Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    return infer_type(self).addcmul(self, tensor1, tensor2);
}
static inline Tensor & addcdiv_out(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    return infer_type(self).addcdiv_out(self, value, tensor1, tensor2, result);
}
static inline Tensor addcdiv(const Tensor & self, Scalar value, const Tensor & tensor1, const Tensor & tensor2) {
    return infer_type(self).addcdiv(self, value, tensor1, tensor2);
}
static inline Tensor & addcdiv_out(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Tensor & result) {
    return infer_type(self).addcdiv_out(self, tensor1, tensor2, result);
}
static inline Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2) {
    return infer_type(self).addcdiv(self, tensor1, tensor2);
}
static inline std::tuple<Tensor &,Tensor &> gesv_out(const Tensor & self, const Tensor & A, Tensor & solution, Tensor & lu) {
    return infer_type(self).gesv_out(self, A, solution, lu);
}
static inline std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) {
    return infer_type(self).gesv(self, A);
}
static inline std::tuple<Tensor &,Tensor &> gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    return infer_type(self).gels_out(self, A, res1, res2);
}
static inline std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) {
    return infer_type(self).gels(self, A);
}
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular, Tensor & res1, Tensor & res2) {
    return infer_type(self).trtrs_out(self, A, upper, transpose, unitriangular, res1, res2);
}
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
    return infer_type(self).trtrs(self, A, upper, transpose, unitriangular);
}
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, bool transpose, Tensor & res1, Tensor & res2) {
    return infer_type(self).trtrs_out(self, A, upper, transpose, res1, res2);
}
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose) {
    return infer_type(self).trtrs(self, A, upper, transpose);
}
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, bool upper, Tensor & res1, Tensor & res2) {
    return infer_type(self).trtrs_out(self, A, upper, res1, res2);
}
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper) {
    return infer_type(self).trtrs(self, A, upper);
}
static inline std::tuple<Tensor &,Tensor &> trtrs_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2) {
    return infer_type(self).trtrs_out(self, A, res1, res2);
}
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A) {
    return infer_type(self).trtrs(self, A);
}
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, bool upper, Tensor & res1, Tensor & res2) {
    return infer_type(self).symeig_out(self, eigenvectors, upper, res1, res2);
}
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
    return infer_type(self).symeig(self, eigenvectors, upper);
}
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) {
    return infer_type(self).symeig_out(self, eigenvectors, res1, res2);
}
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors) {
    return infer_type(self).symeig(self, eigenvectors);
}
static inline std::tuple<Tensor &,Tensor &> symeig_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    return infer_type(self).symeig_out(self, res1, res2);
}
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self) {
    return infer_type(self).symeig(self);
}
static inline std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, bool eigenvectors, Tensor & res1, Tensor & res2) {
    return infer_type(self).eig_out(self, eigenvectors, res1, res2);
}
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
    return infer_type(self).eig(self, eigenvectors);
}
static inline std::tuple<Tensor &,Tensor &> eig_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    return infer_type(self).eig_out(self, res1, res2);
}
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self) {
    return infer_type(self).eig(self);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, bool some, Tensor & res1, Tensor & res2, Tensor & res3) {
    return infer_type(self).svd_out(self, some, res1, res2, res3);
}
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some) {
    return infer_type(self).svd(self, some);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(const Tensor & self, Tensor & res1, Tensor & res2, Tensor & res3) {
    return infer_type(self).svd_out(self, res1, res2, res3);
}
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self) {
    return infer_type(self).svd(self);
}
static inline Tensor & inverse_out(const Tensor & self, Tensor & output) {
    return infer_type(self).inverse_out(self, output);
}
static inline Tensor inverse(const Tensor & self) {
    return infer_type(self).inverse(self);
}
static inline Tensor & potrf_out(const Tensor & self, bool upper, Tensor & output) {
    return infer_type(self).potrf_out(self, upper, output);
}
static inline Tensor potrf(const Tensor & self, bool upper) {
    return infer_type(self).potrf(self, upper);
}
static inline Tensor & potrf_out(const Tensor & self, Tensor & output) {
    return infer_type(self).potrf_out(self, output);
}
static inline Tensor potrf(const Tensor & self) {
    return infer_type(self).potrf(self);
}
static inline Tensor & potrs_out(const Tensor & self, const Tensor & input2, bool upper, Tensor & result) {
    return infer_type(self).potrs_out(self, input2, upper, result);
}
static inline Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) {
    return infer_type(self).potrs(self, input2, upper);
}
static inline Tensor & potrs_out(const Tensor & self, const Tensor & input2, Tensor & result) {
    return infer_type(self).potrs_out(self, input2, result);
}
static inline Tensor potrs(const Tensor & self, const Tensor & input2) {
    return infer_type(self).potrs(self, input2);
}
static inline Tensor & potri_out(const Tensor & self, bool upper, Tensor & output) {
    return infer_type(self).potri_out(self, upper, output);
}
static inline Tensor potri(const Tensor & self, bool upper) {
    return infer_type(self).potri(self, upper);
}
static inline Tensor & potri_out(const Tensor & self, Tensor & output) {
    return infer_type(self).potri_out(self, output);
}
static inline Tensor potri(const Tensor & self) {
    return infer_type(self).potri(self);
}
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Scalar tol, Tensor & res1, Tensor & res2) {
    return infer_type(self).pstrf_out(self, upper, tol, res1, res2);
}
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) {
    return infer_type(self).pstrf(self, upper, tol);
}
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, bool upper, Tensor & res1, Tensor & res2) {
    return infer_type(self).pstrf_out(self, upper, res1, res2);
}
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper) {
    return infer_type(self).pstrf(self, upper);
}
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Scalar tol, Tensor & res1, Tensor & res2) {
    return infer_type(self).pstrf_out(self, tol, res1, res2);
}
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, Scalar tol) {
    return infer_type(self).pstrf(self, tol);
}
static inline std::tuple<Tensor &,Tensor &> pstrf_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    return infer_type(self).pstrf_out(self, res1, res2);
}
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self) {
    return infer_type(self).pstrf(self);
}
static inline std::tuple<Tensor &,Tensor &> qr_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    return infer_type(self).qr_out(self, res1, res2);
}
static inline std::tuple<Tensor,Tensor> qr(const Tensor & self) {
    return infer_type(self).qr(self);
}
static inline std::tuple<Tensor &,Tensor &> geqrf_out(const Tensor & self, Tensor & res1, Tensor & res2) {
    return infer_type(self).geqrf_out(self, res1, res2);
}
static inline std::tuple<Tensor,Tensor> geqrf(const Tensor & self) {
    return infer_type(self).geqrf(self);
}
static inline std::tuple<Tensor &,const Tensor &> orgqr_out(const Tensor & self, const Tensor & input2, Tensor & result) {
    return infer_type(self).orgqr_out(self, input2, result);
}
static inline std::tuple<Tensor,const Tensor &> orgqr(const Tensor & self, const Tensor & input2) {
    return infer_type(self).orgqr(self, input2);
}
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose, Tensor & result) {
    return infer_type(self).ormqr_out(self, input2, input3, left, transpose, result);
}
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    return infer_type(self).ormqr(self, input2, input3, left, transpose);
}
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, Tensor & result) {
    return infer_type(self).ormqr_out(self, input2, input3, left, result);
}
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left) {
    return infer_type(self).ormqr(self, input2, input3, left);
}
static inline std::tuple<Tensor &,const Tensor &> ormqr_out(const Tensor & self, const Tensor & input2, const Tensor & input3, Tensor & result) {
    return infer_type(self).ormqr_out(self, input2, input3, result);
}
static inline std::tuple<Tensor,const Tensor &> ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3) {
    return infer_type(self).ormqr(self, input2, input3);
}
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) {
    return infer_type(self).btrifact_out(info, pivot, self, result, pivots);
}
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & info, bool pivot, const Tensor & self) {
    return infer_type(self).btrifact(info, pivot, self);
}
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & info, const Tensor & self, Tensor & result, Tensor & pivots) {
    return infer_type(self).btrifact_out(info, self, result, pivots);
}
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & info, const Tensor & self) {
    return infer_type(self).btrifact(info, self);
}
static inline std::tuple<Tensor &,Tensor &> btrifact_out(bool pivot, const Tensor & self, Tensor & result, Tensor & pivots) {
    return infer_type(self).btrifact_out(pivot, self, result, pivots);
}
static inline std::tuple<Tensor,Tensor> btrifact(bool pivot, const Tensor & self) {
    return infer_type(self).btrifact(pivot, self);
}
static inline std::tuple<Tensor &,Tensor &> btrifact_out(const Tensor & self, Tensor & result, Tensor & pivots) {
    return infer_type(self).btrifact_out(self, result, pivots);
}
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self) {
    return infer_type(self).btrifact(self);
}
static inline Tensor & btrisolve_out(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots, Tensor & result) {
    return infer_type(self).btrisolve_out(self, LU_data, LU_pivots, result);
}
static inline Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
    return infer_type(self).btrisolve(self, LU_data, LU_pivots);
}
static inline Tensor & randperm_out(Generator & generator, int64_t n, Tensor & result) {
    return infer_type(result).randperm_out(generator, n, result);
}
static inline Tensor & randperm_out(int64_t n, Tensor & result) {
    return infer_type(result).randperm_out(n, result);
}
static inline Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) {
    return infer_type(self).multinomial_out(generator, self, num_samples, replacement, result);
}
static inline Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples, bool replacement) {
    return infer_type(self).multinomial(generator, self, num_samples, replacement);
}
static inline Tensor & multinomial_out(Generator & generator, const Tensor & self, int64_t num_samples, Tensor & result) {
    return infer_type(self).multinomial_out(generator, self, num_samples, result);
}
static inline Tensor multinomial(Generator & generator, const Tensor & self, int64_t num_samples) {
    return infer_type(self).multinomial(generator, self, num_samples);
}
static inline Tensor & multinomial_out(const Tensor & self, int64_t num_samples, bool replacement, Tensor & result) {
    return infer_type(self).multinomial_out(self, num_samples, replacement, result);
}
static inline Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement) {
    return infer_type(self).multinomial(self, num_samples, replacement);
}
static inline Tensor & multinomial_out(const Tensor & self, int64_t num_samples, Tensor & result) {
    return infer_type(self).multinomial_out(self, num_samples, result);
}
static inline Tensor multinomial(const Tensor & self, int64_t num_samples) {
    return infer_type(self).multinomial(self, num_samples);
}
static inline Tensor & rand_out(Generator & generator, IntList size, Tensor & result) {
    return infer_type(result).rand_out(generator, size, result);
}
static inline Tensor & rand_out(IntList size, Tensor & result) {
    return infer_type(result).rand_out(size, result);
}
static inline Tensor & randn_out(Generator & generator, IntList size, Tensor & result) {
    return infer_type(result).randn_out(generator, size, result);
}
static inline Tensor & randn_out(IntList size, Tensor & result) {
    return infer_type(result).randn_out(size, result);
}
static inline Tensor & select_out(const Tensor & self, int dim, int64_t sliceIndex, Tensor & result) {
    return infer_type(self).select_out(self, dim, sliceIndex, result);
}
static inline Tensor select(const Tensor & self, int dim, int64_t sliceIndex) {
    return infer_type(self).select(self, dim, sliceIndex);
}
static inline Tensor & cat_out(TensorList tensors, int dim, Tensor & self) {
    return infer_type(tensors).cat_out(tensors, dim, self);
}
static inline Tensor cat(TensorList tensors, int dim) {
    return infer_type(tensors).cat(tensors, dim);
}
static inline void Abs_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).Abs_updateOutput(input, output);
}
static inline void Abs_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    return infer_type(input).Abs_updateGradInput(input, gradOutput, gradInput);
}
static inline void AbsCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).AbsCriterion_updateOutput(input, target, output, sizeAverage);
}
static inline void AbsCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).AbsCriterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights) {
    return infer_type(input).BCECriterion_updateOutput(input, target, output, sizeAverage, weights);
}
static inline void BCECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).BCECriterion_updateOutput(input, target, output, sizeAverage);
}
static inline void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights) {
    return infer_type(input).BCECriterion_updateGradInput(input, target, gradInput, sizeAverage, weights);
}
static inline void BCECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).BCECriterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).ClassNLLCriterion_updateOutput(input, target, output, sizeAverage, weights, total_weight, ignore_index);
}
static inline void ClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).ClassNLLCriterion_updateOutput(input, target, output, sizeAverage, total_weight, ignore_index);
}
static inline void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).ClassNLLCriterion_updateGradInput(input, target, gradInput, sizeAverage, weights, total_weight, ignore_index);
}
static inline void ClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).ClassNLLCriterion_updateGradInput(input, target, gradInput, sizeAverage, total_weight, ignore_index);
}
static inline void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).SpatialClassNLLCriterion_updateOutput(input, target, output, sizeAverage, weights, total_weight, ignore_index);
}
static inline void SpatialClassNLLCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).SpatialClassNLLCriterion_updateOutput(input, target, output, sizeAverage, total_weight, ignore_index);
}
static inline void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & weights, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).SpatialClassNLLCriterion_updateGradInput(input, target, gradInput, sizeAverage, weights, total_weight, ignore_index);
}
static inline void SpatialClassNLLCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, const Tensor & total_weight, int64_t ignore_index) {
    return infer_type(input).SpatialClassNLLCriterion_updateGradInput(input, target, gradInput, sizeAverage, total_weight, ignore_index);
}
static inline void ELU_updateOutput(const Tensor & input, const Tensor & output, Scalar alpha, bool inplace) {
    return infer_type(input).ELU_updateOutput(input, output, alpha, inplace);
}
static inline void ELU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar alpha, bool inplace) {
    return infer_type(input).ELU_updateGradInput(input, gradOutput, gradInput, output, alpha, inplace);
}
static inline void DistKLDivCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).DistKLDivCriterion_updateOutput(input, target, output, sizeAverage);
}
static inline void DistKLDivCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).DistKLDivCriterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void GatedLinear_updateOutput(const Tensor & input, const Tensor & output, int dim) {
    return infer_type(input).GatedLinear_updateOutput(input, output, dim);
}
static inline void GatedLinear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int dim) {
    return infer_type(input).GatedLinear_updateGradInput(input, gradOutput, gradInput, dim);
}
static inline void HardShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) {
    return infer_type(input).HardShrink_updateOutput(input, output, lambda);
}
static inline void HardShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) {
    return infer_type(input).HardShrink_updateGradInput(input, gradOutput, gradInput, lambda);
}
static inline void HardTanh_updateOutput(const Tensor & input, const Tensor & output, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(input).HardTanh_updateOutput(input, output, min_val, max_val, inplace);
}
static inline void HardTanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(input).HardTanh_updateGradInput(input, gradOutput, gradInput, min_val, max_val, inplace);
}
static inline void L1Cost_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).L1Cost_updateOutput(input, output);
}
static inline void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    return infer_type(input).L1Cost_updateGradInput(input, gradOutput, gradInput);
}
static inline void L1Cost_updateGradInput(const Tensor & input, const Tensor & gradInput) {
    return infer_type(input).L1Cost_updateGradInput(input, gradInput);
}
static inline void LeakyReLU_updateOutput(const Tensor & input, const Tensor & output, Scalar negval, bool inplace) {
    return infer_type(input).LeakyReLU_updateOutput(input, output, negval, inplace);
}
static inline void LeakyReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar negval, bool inplace) {
    return infer_type(input).LeakyReLU_updateGradInput(input, gradOutput, gradInput, negval, inplace);
}
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    return infer_type(input).GRUFused_updateOutput(input, hidden, bias1, bias2, hx, output, storage);
}
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    return infer_type(input).GRUFused_updateOutput(input, hidden, bias1, hx, output, storage);
}
static inline void GRUFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & hx, const Tensor & output, const Tensor & storage) {
    return infer_type(input).GRUFused_updateOutput(input, hidden, hx, output, storage);
}
static inline void GRUFused_updateGradInput(const Tensor & gradInInput, const Tensor & gradInHidden, const Tensor & gradOutput, const Tensor & gradInputHx, const Tensor & storage) {
    return infer_type(gradInInput).GRUFused_updateGradInput(gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
}
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & bias2, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    return infer_type(input).LSTMFused_updateOutput(input, hidden, bias1, bias2, cell, output, outputCell);
}
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & bias1, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    return infer_type(input).LSTMFused_updateOutput(input, hidden, bias1, cell, output, outputCell);
}
static inline void LSTMFused_updateOutput(const Tensor & input, const Tensor & hidden, const Tensor & cell, const Tensor & output, const Tensor & outputCell) {
    return infer_type(input).LSTMFused_updateOutput(input, hidden, cell, output, outputCell);
}
static inline void LSTMFused_updateGradInput(const Tensor & storage, const Tensor & gradInGates, const Tensor & cx, const Tensor & cy, const Tensor & gradOutput, const Tensor & gradOutputCell, const Tensor & gradInputCx) {
    return infer_type(storage).LSTMFused_updateGradInput(storage, gradInGates, cx, cy, gradOutput, gradOutputCell, gradInputCx);
}
static inline void LogSigmoid_updateOutput(const Tensor & input, const Tensor & output, const Tensor & buffer) {
    return infer_type(input).LogSigmoid_updateOutput(input, output, buffer);
}
static inline void LogSigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & buffer) {
    return infer_type(input).LogSigmoid_updateGradInput(input, gradOutput, gradInput, buffer);
}
static inline void LogSoftMax_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).LogSoftMax_updateOutput(input, output);
}
static inline void LogSoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(input).LogSoftMax_updateGradInput(input, gradOutput, gradInput, output);
}
static inline void MarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, Scalar margin) {
    return infer_type(input).MarginCriterion_updateOutput(input, target, output, sizeAverage, margin);
}
static inline void MarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, Scalar margin) {
    return infer_type(input).MarginCriterion_updateGradInput(input, target, gradInput, sizeAverage, margin);
}
static inline void SoftMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).SoftMarginCriterion_updateOutput(input, target, output, sizeAverage);
}
static inline void SoftMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).SoftMarginCriterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void MSECriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).MSECriterion_updateOutput(input, target, output, sizeAverage);
}
static inline void MSECriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).MSECriterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void MultiLabelMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, const Tensor & isTarget, bool sizeAverage) {
    return infer_type(input).MultiLabelMarginCriterion_updateOutput(input, target, output, isTarget, sizeAverage);
}
static inline void MultiLabelMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, const Tensor & isTarget, bool sizeAverage) {
    return infer_type(input).MultiLabelMarginCriterion_updateGradInput(input, target, gradInput, isTarget, sizeAverage);
}
static inline void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, const Tensor & weights, Scalar margin) {
    return infer_type(input).MultiMarginCriterion_updateOutput(input, target, output, sizeAverage, p, weights, margin);
}
static inline void MultiMarginCriterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage, int p, Scalar margin) {
    return infer_type(input).MultiMarginCriterion_updateOutput(input, target, output, sizeAverage, p, margin);
}
static inline void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, const Tensor & weights, Scalar margin) {
    return infer_type(input).MultiMarginCriterion_updateGradInput(input, target, gradInput, sizeAverage, p, weights, margin);
}
static inline void MultiMarginCriterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage, int p, Scalar margin) {
    return infer_type(input).MultiMarginCriterion_updateGradInput(input, target, gradInput, sizeAverage, p, margin);
}
static inline void PReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, int64_t nOutputPlane) {
    return infer_type(input).PReLU_updateOutput(input, output, weight, nOutputPlane);
}
static inline void PReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int64_t nOutputPlane) {
    return infer_type(input).PReLU_updateGradInput(input, gradOutput, gradInput, weight, nOutputPlane);
}
static inline void PReLU_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradWeight, const Tensor & gradWeightBuf, const Tensor & gradWeightBuf2, int64_t nOutputPlane, Scalar scale) {
    return infer_type(input).PReLU_accGradParameters(input, gradOutput, gradInput, weight, gradWeight, gradWeightBuf, gradWeightBuf2, nOutputPlane, scale);
}
static inline void Linear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & addBuffer) {
    return infer_type(input).Linear_updateOutput(input, output, weight, bias, addBuffer);
}
static inline void Linear_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight) {
    return infer_type(input).Linear_updateGradInput(input, gradOutput, gradInput, weight);
}
static inline void Linear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & addBuffer, Scalar scale) {
    return infer_type(input).Linear_accGradParameters(input, gradOutput, gradInput, weight, bias, gradWeight, gradBias, addBuffer, scale);
}
static inline void RReLU_updateOutput(const Tensor & input, const Tensor & output, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace, Generator & generator) {
    return infer_type(input).RReLU_updateOutput(input, output, noise, lower, upper, train, inplace, generator);
}
static inline void RReLU_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & noise, Scalar lower, Scalar upper, bool train, bool inplace) {
    return infer_type(input).RReLU_updateGradInput(input, gradOutput, gradInput, noise, lower, upper, train, inplace);
}
static inline void Sigmoid_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).Sigmoid_updateOutput(input, output);
}
static inline void Sigmoid_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(input).Sigmoid_updateGradInput(input, gradOutput, gradInput, output);
}
static inline void Sigmoid_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(gradOutput).Sigmoid_updateGradInput(gradOutput, gradInput, output);
}
static inline void SmoothL1Criterion_updateOutput(const Tensor & input, const Tensor & target, const Tensor & output, bool sizeAverage) {
    return infer_type(input).SmoothL1Criterion_updateOutput(input, target, output, sizeAverage);
}
static inline void SmoothL1Criterion_updateGradInput(const Tensor & input, const Tensor & target, const Tensor & gradInput, bool sizeAverage) {
    return infer_type(input).SmoothL1Criterion_updateGradInput(input, target, gradInput, sizeAverage);
}
static inline void SoftMax_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).SoftMax_updateOutput(input, output);
}
static inline void SoftMax_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(input).SoftMax_updateGradInput(input, gradOutput, gradInput, output);
}
static inline void SoftPlus_updateOutput(const Tensor & input, const Tensor & output, Scalar beta, Scalar threshold) {
    return infer_type(input).SoftPlus_updateOutput(input, output, beta, threshold);
}
static inline void SoftPlus_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output, Scalar beta, Scalar threshold) {
    return infer_type(input).SoftPlus_updateGradInput(input, gradOutput, gradInput, output, beta, threshold);
}
static inline void SoftShrink_updateOutput(const Tensor & input, const Tensor & output, Scalar lambda) {
    return infer_type(input).SoftShrink_updateOutput(input, output, lambda);
}
static inline void SoftShrink_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar lambda) {
    return infer_type(input).SoftShrink_updateGradInput(input, gradOutput, gradInput, lambda);
}
static inline void IndexLinear_updateOutput(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & normalizedValues, int train) {
    return infer_type(values).IndexLinear_updateOutput(keys, keysOffset, values, sizes, cumSumSizes, output, weight, bias, normalizedValues, train);
}
static inline void IndexLinear_accGradParameters(const Tensor & keys, int64_t keysOffset, const Tensor & values, const Tensor & sizes, const Tensor & cumSumSizes, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, const Tensor & valuesBuffer, Scalar weightDecay, Scalar scale) {
    return infer_type(values).IndexLinear_accGradParameters(keys, keysOffset, values, sizes, cumSumSizes, gradOutput, gradWeight, gradBias, weight, bias, valuesBuffer, weightDecay, scale);
}
static inline void SparseLinear_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias) {
    return infer_type(input).SparseLinear_updateOutput(input, output, weight, bias);
}
static inline void SparseLinear_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & bias, Scalar weightDecay, Scalar scale) {
    return infer_type(input).SparseLinear_accGradParameters(input, gradOutput, gradWeight, gradBias, weight, bias, weightDecay, scale);
}
static inline void Sqrt_updateOutput(const Tensor & input, const Tensor & output, Scalar eps) {
    return infer_type(input).Sqrt_updateOutput(input, output, eps);
}
static inline void Sqrt_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(input).Sqrt_updateGradInput(input, gradOutput, gradInput, output);
}
static inline void Square_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).Square_updateOutput(input, output);
}
static inline void Square_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    return infer_type(input).Square_updateGradInput(input, gradOutput, gradInput);
}
static inline void Tanh_updateOutput(const Tensor & input, const Tensor & output) {
    return infer_type(input).Tanh_updateOutput(input, output);
}
static inline void Tanh_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(input).Tanh_updateGradInput(input, gradOutput, gradInput, output);
}
static inline void Tanh_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, const Tensor & output) {
    return infer_type(gradOutput).Tanh_updateGradInput(gradOutput, gradInput, output);
}
static inline void Threshold_updateOutput(const Tensor & input, const Tensor & output, Scalar threshold, Scalar val, bool inplace) {
    return infer_type(input).Threshold_updateOutput(input, output, threshold, val, inplace);
}
static inline void Threshold_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, Scalar threshold, Scalar val, bool inplace) {
    return infer_type(input).Threshold_updateGradInput(input, gradOutput, gradInput, threshold, val, inplace);
}
static inline void TemporalConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize, int outputFrameSize) {
    return infer_type(input).TemporalConvolution_updateOutput(input, output, weight, bias, kW, dW, inputFrameSize, outputFrameSize);
}
static inline void TemporalConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) {
    return infer_type(input).TemporalConvolution_updateGradInput(input, gradOutput, gradInput, weight, kW, dW);
}
static inline void TemporalConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) {
    return infer_type(input).TemporalConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, kW, dW, scale);
}
static inline void TemporalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int dW) {
    return infer_type(input).TemporalMaxPooling_updateOutput(input, output, indices, kW, dW);
}
static inline void TemporalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int dW) {
    return infer_type(input).TemporalMaxPooling_updateGradInput(input, gradOutput, gradInput, indices, kW, dW);
}
static inline void TemporalSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int dW, int inputFrameSize) {
    return infer_type(input).TemporalSubSampling_updateOutput(input, output, weight, bias, kW, dW, inputFrameSize);
}
static inline void TemporalSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int dW) {
    return infer_type(input).TemporalSubSampling_updateGradInput(input, gradOutput, gradInput, weight, kW, dW);
}
static inline void TemporalSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int dW, Scalar scale) {
    return infer_type(input).TemporalSubSampling_accGradParameters(input, gradOutput, gradWeight, gradBias, kW, dW, scale);
}
static inline void TemporalRowConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) {
    return infer_type(input).TemporalRowConvolution_updateOutput(input, output, weight, bias, finput, fgradInput, kW, dW, padW, featFirst);
}
static inline void TemporalRowConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst) {
    return infer_type(input).TemporalRowConvolution_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, kW, dW, padW, featFirst);
}
static inline void TemporalRowConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int dW, int padW, bool featFirst, Scalar scale) {
    return infer_type(input).TemporalRowConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, kW, dW, padW, featFirst, scale);
}
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    return infer_type(input).BatchNormalization_updateOutput(input, output, weight, bias, running_mean, running_var, save_mean, save_std, train, momentum, eps);
}
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    return infer_type(input).BatchNormalization_updateOutput(input, output, weight, running_mean, running_var, save_mean, save_std, train, momentum, eps);
}
static inline void BatchNormalization_updateOutput(const Tensor & input, const Tensor & output, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double momentum, double eps) {
    return infer_type(input).BatchNormalization_updateOutput(input, output, running_mean, running_var, save_mean, save_std, train, momentum, eps);
}
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    return infer_type(input).BatchNormalization_backward(input, gradOutput, gradInput, gradWeight, gradBias, weight, running_mean, running_var, save_mean, save_std, train, scale, eps);
}
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    return infer_type(input).BatchNormalization_backward(input, gradOutput, gradInput, gradWeight, gradBias, running_mean, running_var, save_mean, save_std, train, scale, eps);
}
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & gradWeight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    return infer_type(input).BatchNormalization_backward(input, gradOutput, gradInput, gradWeight, running_mean, running_var, save_mean, save_std, train, scale, eps);
}
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    return infer_type(input).BatchNormalization_backward(input, gradOutput, gradInput, running_mean, running_var, save_mean, save_std, train, scale, eps);
}
static inline void BatchNormalization_backward(const Tensor & input, const Tensor & gradOutput, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_std, bool train, double scale, double eps) {
    return infer_type(input).BatchNormalization_backward(input, gradOutput, running_mean, running_var, save_mean, save_std, train, scale, eps);
}
static inline void SpatialConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    return infer_type(input).SpatialConvolutionMap_updateOutput(input, output, weight, bias, connTable, nInputPlane, nOutputPlane, dW, dH);
}
static inline void SpatialConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    return infer_type(input).SpatialConvolutionMap_updateGradInput(input, gradOutput, gradInput, weight, bias, connTable, nInputPlane, nOutputPlane, dW, dH);
}
static inline void SpatialConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) {
    return infer_type(input).SpatialConvolutionMap_accGradParameters(input, gradOutput, gradWeight, gradBias, connTable, nInputPlane, nOutputPlane, dW, dH, scale);
}
static inline void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialConvolutionMM_updateOutput(input, output, weight, bias, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialConvolutionMM_updateOutput(input, output, weight, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialConvolutionMM_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    return infer_type(input).SpatialConvolutionMM_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, kW, kH, dW, dH, padW, padH, scale);
}
static inline void SpatialConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    return infer_type(input).SpatialConvolutionMM_accGradParameters(input, gradOutput, gradWeight, finput, fgradInput, kW, kH, dW, dH, padW, padH, scale);
}
static inline void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialDepthWiseConvolution_updateOutput(input, output, weight, bias, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialDepthWiseConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialDepthWiseConvolution_updateOutput(input, output, weight, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialDepthWiseConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH) {
    return infer_type(input).SpatialDepthWiseConvolution_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, kW, kH, dW, dH, padW, padH);
}
static inline void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    return infer_type(input).SpatialDepthWiseConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, kW, kH, dW, dH, padW, padH, scale);
}
static inline void SpatialDepthWiseConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, Scalar scale) {
    return infer_type(input).SpatialDepthWiseConvolution_accGradParameters(input, gradOutput, gradWeight, finput, fgradInput, kW, kH, dW, dH, padW, padH, scale);
}
static inline void SpatialConvolutionLocal_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) {
    return infer_type(input).SpatialConvolutionLocal_updateOutput(input, output, weight, bias, finput, fgradInput, kW, kH, dW, dH, padW, padH, inputWidth, inputHeight, outputWidth, outputHeight);
}
static inline void SpatialConvolutionLocal_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight) {
    return infer_type(input).SpatialConvolutionLocal_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, kW, kH, dW, dH, padW, padH, inputWidth, inputHeight, outputWidth, outputHeight);
}
static inline void SpatialConvolutionLocal_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int kW, int kH, int dW, int dH, int padW, int padH, int64_t inputWidth, int64_t inputHeight, int64_t outputWidth, int64_t outputHeight, Scalar scale) {
    return infer_type(input).SpatialConvolutionLocal_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, kW, kH, dW, dH, padW, padH, inputWidth, inputHeight, outputWidth, outputHeight, scale);
}
static inline void SpatialAdaptiveMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) {
    return infer_type(input).SpatialAdaptiveMaxPooling_updateOutput(input, output, indices, owidth, oheight);
}
static inline void SpatialAdaptiveMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices) {
    return infer_type(input).SpatialAdaptiveMaxPooling_updateGradInput(input, gradOutput, gradInput, indices);
}
static inline void SpatialAdaptiveAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int owidth, int oheight) {
    return infer_type(input).SpatialAdaptiveAveragePooling_updateOutput(input, output, owidth, oheight);
}
static inline void SpatialAdaptiveAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput) {
    return infer_type(input).SpatialAdaptiveAveragePooling_updateGradInput(input, gradOutput, gradInput);
}
static inline void SpatialAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).SpatialAveragePooling_updateOutput(input, output, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}
static inline void SpatialAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).SpatialAveragePooling_updateGradInput(input, gradOutput, gradInput, kW, kH, dW, dH, padW, padH, ceil_mode, count_include_pad);
}
static inline void SpatialFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) {
    return infer_type(input).SpatialFractionalMaxPooling_updateOutput(input, output, outputW, outputH, poolSizeW, poolSizeH, indices, randomSamples);
}
static inline void SpatialFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputW, int outputH, int poolSizeW, int poolSizeH, const Tensor & indices) {
    return infer_type(input).SpatialFractionalMaxPooling_updateGradInput(input, gradOutput, gradInput, outputW, outputH, poolSizeW, poolSizeH, indices);
}
static inline void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    return infer_type(input).SpatialFullConvolution_updateOutput(input, output, weight, bias, columns, ones, kW, kH, dW, dH, padW, padH, adjW, adjH);
}
static inline void SpatialFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    return infer_type(input).SpatialFullConvolution_updateOutput(input, output, weight, columns, ones, kW, kH, dW, dH, padW, padH, adjW, adjH);
}
static inline void SpatialFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH) {
    return infer_type(input).SpatialFullConvolution_updateGradInput(input, gradOutput, gradInput, weight, gradColumns, kW, kH, dW, dH, padW, padH, adjW, adjH);
}
static inline void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) {
    return infer_type(input).SpatialFullConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, columns, ones, kW, kH, dW, dH, padW, padH, adjW, adjH, scale);
}
static inline void SpatialFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int adjW, int adjH, Scalar scale) {
    return infer_type(input).SpatialFullConvolution_accGradParameters(input, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH, padW, padH, adjW, adjH, scale);
}
static inline void SpatialFullConvolutionMap_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    return infer_type(input).SpatialFullConvolutionMap_updateOutput(input, output, weight, bias, connTable, nInputPlane, nOutputPlane, dW, dH);
}
static inline void SpatialFullConvolutionMap_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & bias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH) {
    return infer_type(input).SpatialFullConvolutionMap_updateGradInput(input, gradOutput, gradInput, weight, bias, connTable, nInputPlane, nOutputPlane, dW, dH);
}
static inline void SpatialFullConvolutionMap_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & connTable, int nInputPlane, int nOutputPlane, int dW, int dH, Scalar scale) {
    return infer_type(input).SpatialFullConvolutionMap_accGradParameters(input, gradOutput, gradWeight, gradBias, connTable, nInputPlane, nOutputPlane, dW, dH, scale);
}
static inline void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    return infer_type(input).SpatialDilatedConvolution_updateOutput(input, output, weight, bias, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
}
static inline void SpatialDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    return infer_type(input).SpatialDilatedConvolution_updateOutput(input, output, weight, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
}
static inline void SpatialDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH) {
    return infer_type(input).SpatialDilatedConvolution_updateGradInput(input, gradOutput, gradInput, weight, gradColumns, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
}
static inline void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) {
    return infer_type(input).SpatialDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, scale);
}
static inline void SpatialDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, Scalar scale) {
    return infer_type(input).SpatialDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, scale);
}
static inline void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) {
    return infer_type(input).SpatialFullDilatedConvolution_updateOutput(input, output, weight, bias, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
}
static inline void SpatialFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) {
    return infer_type(input).SpatialFullDilatedConvolution_updateOutput(input, output, weight, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
}
static inline void SpatialFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH) {
    return infer_type(input).SpatialFullDilatedConvolution_updateGradInput(input, gradOutput, gradInput, weight, gradColumns, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH);
}
static inline void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale) {
    return infer_type(input).SpatialFullDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH, scale);
}
static inline void SpatialFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, int adjW, int adjH, Scalar scale) {
    return infer_type(input).SpatialFullDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, columns, ones, kW, kH, dW, dH, padW, padH, dilationW, dilationH, adjW, adjH, scale);
}
static inline void SpatialMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) {
    return infer_type(input).SpatialMaxPooling_updateOutput(input, output, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}
static inline void SpatialMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, bool ceil_mode) {
    return infer_type(input).SpatialMaxPooling_updateGradInput(input, gradOutput, gradInput, indices, kW, kH, dW, dH, padW, padH, ceil_mode);
}
static inline void SpatialDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) {
    return infer_type(input).SpatialDilatedMaxPooling_updateOutput(input, output, indices, kW, kH, dW, dH, padW, padH, dilationW, dilationH, ceil_mode);
}
static inline void SpatialDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kW, int kH, int dW, int dH, int padW, int padH, int dilationW, int dilationH, bool ceil_mode) {
    return infer_type(input).SpatialDilatedMaxPooling_updateGradInput(input, gradOutput, gradInput, indices, kW, kH, dW, dH, padW, padH, dilationW, dilationH, ceil_mode);
}
static inline void SpatialMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int owidth, int oheight) {
    return infer_type(input).SpatialMaxUnpooling_updateOutput(input, output, indices, owidth, oheight);
}
static inline void SpatialMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int owidth, int oheight) {
    return infer_type(input).SpatialMaxUnpooling_updateGradInput(input, gradOutput, gradInput, indices, owidth, oheight);
}
static inline void SpatialSubSampling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, int kW, int kH, int dW, int dH) {
    return infer_type(input).SpatialSubSampling_updateOutput(input, output, weight, bias, kW, kH, dW, dH);
}
static inline void SpatialSubSampling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, int kW, int kH, int dW, int dH) {
    return infer_type(input).SpatialSubSampling_updateGradInput(input, gradOutput, gradInput, weight, kW, kH, dW, dH);
}
static inline void SpatialSubSampling_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, int kW, int kH, int dW, int dH, Scalar scale) {
    return infer_type(input).SpatialSubSampling_accGradParameters(input, gradOutput, gradWeight, gradBias, kW, kH, dW, dH, scale);
}
static inline void SpatialUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) {
    return infer_type(input).SpatialUpSamplingNearest_updateOutput(input, output, scale_factor);
}
static inline void SpatialUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) {
    return infer_type(input).SpatialUpSamplingNearest_updateGradInput(input, gradOutput, gradInput, scale_factor);
}
static inline void SpatialUpSamplingBilinear_updateOutput(const Tensor & input, const Tensor & output, int outputHeight, int outputWidth) {
    return infer_type(input).SpatialUpSamplingBilinear_updateOutput(input, output, outputHeight, outputWidth);
}
static inline void SpatialUpSamplingBilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputHeight, int inputWidth, int outputHeight, int outputWidth) {
    return infer_type(gradOutput).SpatialUpSamplingBilinear_updateGradInput(gradOutput, gradInput, nbatch, nchannels, inputHeight, inputWidth, outputHeight, outputWidth);
}
static inline void SpatialGridSamplerBilinear_updateOutput(const Tensor & input, const Tensor & grid, const Tensor & output) {
    return infer_type(input).SpatialGridSamplerBilinear_updateOutput(input, grid, output);
}
static inline void SpatialGridSamplerBilinear_updateGradInput(const Tensor & input, const Tensor & gradInput, const Tensor & grid, const Tensor & gradGrid, const Tensor & gradOutput) {
    return infer_type(input).SpatialGridSamplerBilinear_updateGradInput(input, gradInput, grid, gradGrid, gradOutput);
}
static inline void VolumetricAveragePooling_updateOutput(const Tensor & input, const Tensor & output, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).VolumetricAveragePooling_updateOutput(input, output, kT, kW, kH, dT, dW, dH, padT, padW, padH, ceil_mode, count_include_pad);
}
static inline void VolumetricAveragePooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).VolumetricAveragePooling_updateGradInput(input, gradOutput, gradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH, ceil_mode, count_include_pad);
}
static inline void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolution_updateOutput(input, output, weight, bias, finput, fgradInput, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolution_updateOutput(input, output, weight, finput, fgradInput, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolution_updateGradInput(input, gradOutput, gradInput, weight, finput, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    return infer_type(input).VolumetricConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, dT, dW, dH, pT, pW, pH, scale);
}
static inline void VolumetricConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    return infer_type(input).VolumetricConvolution_accGradParameters(input, gradOutput, gradWeight, finput, fgradInput, dT, dW, dH, pT, pW, pH, scale);
}
static inline void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolutionMM_updateOutput(input, output, weight, bias, finput, kT, kW, kH, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolutionMM_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolutionMM_updateOutput(input, output, weight, finput, kT, kW, kH, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolutionMM_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricConvolutionMM_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, kT, kW, kH, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    return infer_type(input).VolumetricConvolutionMM_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, kT, kW, kH, dT, dW, dH, pT, pW, pH, scale);
}
static inline void VolumetricConvolutionMM_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, Scalar scale) {
    return infer_type(input).VolumetricConvolutionMM_accGradParameters(input, gradOutput, gradWeight, finput, kT, kW, kH, dT, dW, dH, pT, pW, pH, scale);
}
static inline void VolumetricFractionalMaxPooling_updateOutput(const Tensor & input, const Tensor & output, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices, const Tensor & randomSamples) {
    return infer_type(input).VolumetricFractionalMaxPooling_updateOutput(input, output, outputT, outputW, outputH, poolSizeT, poolSizeW, poolSizeH, indices, randomSamples);
}
static inline void VolumetricFractionalMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int outputT, int outputW, int outputH, int poolSizeT, int poolSizeW, int poolSizeH, const Tensor & indices) {
    return infer_type(input).VolumetricFractionalMaxPooling_updateGradInput(input, gradOutput, gradInput, outputT, outputW, outputH, poolSizeT, poolSizeW, poolSizeH, indices);
}
static inline void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullConvolution_updateOutput(input, output, weight, bias, finput, fgradInput, dT, dW, dH, pT, pW, pH, aT, aW, aH);
}
static inline void VolumetricFullConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullConvolution_updateOutput(input, output, weight, finput, fgradInput, dT, dW, dH, pT, pW, pH, aT, aW, aH);
}
static inline void VolumetricFullConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullConvolution_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, dT, dW, dH, pT, pW, pH, aT, aW, aH);
}
static inline void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) {
    return infer_type(input).VolumetricFullConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, dT, dW, dH, pT, pW, pH, aT, aW, aH, scale);
}
static inline void VolumetricFullConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int aT, int aW, int aH, Scalar scale) {
    return infer_type(input).VolumetricFullConvolution_accGradParameters(input, gradOutput, gradWeight, finput, fgradInput, dT, dW, dH, pT, pW, pH, aT, aW, aH, scale);
}
static inline void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    return infer_type(input).VolumetricDilatedConvolution_updateOutput(input, output, weight, bias, columns, ones, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
}
static inline void VolumetricDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    return infer_type(input).VolumetricDilatedConvolution_updateOutput(input, output, weight, columns, ones, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
}
static inline void VolumetricDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & gradColumns, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH) {
    return infer_type(input).VolumetricDilatedConvolution_updateGradInput(input, gradOutput, gradInput, weight, gradColumns, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH);
}
static inline void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) {
    return infer_type(input).VolumetricDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, columns, ones, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH, scale);
}
static inline void VolumetricDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & columns, const Tensor & ones, int kT, int kW, int kH, int dT, int dW, int dH, int padT, int padW, int padH, int dilationT, int dilationW, int dilationH, Scalar scale) {
    return infer_type(input).VolumetricDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, columns, ones, kT, kW, kH, dT, dW, dH, padT, padW, padH, dilationT, dilationW, dilationH, scale);
}
static inline void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & bias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullDilatedConvolution_updateOutput(input, output, weight, bias, finput, fgradInput, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
}
static inline void VolumetricFullDilatedConvolution_updateOutput(const Tensor & input, const Tensor & output, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullDilatedConvolution_updateOutput(input, output, weight, finput, fgradInput, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
}
static inline void VolumetricFullDilatedConvolution_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & weight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH) {
    return infer_type(input).VolumetricFullDilatedConvolution_updateGradInput(input, gradOutput, gradInput, weight, finput, fgradInput, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH);
}
static inline void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & gradBias, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale) {
    return infer_type(input).VolumetricFullDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, gradBias, finput, fgradInput, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, scale);
}
static inline void VolumetricFullDilatedConvolution_accGradParameters(const Tensor & input, const Tensor & gradOutput, const Tensor & gradWeight, const Tensor & finput, const Tensor & fgradInput, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, int aT, int aW, int aH, Scalar scale) {
    return infer_type(input).VolumetricFullDilatedConvolution_accGradParameters(input, gradOutput, gradWeight, finput, fgradInput, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, aT, aW, aH, scale);
}
static inline void VolumetricMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) {
    return infer_type(input).VolumetricMaxPooling_updateOutput(input, output, indices, kT, kW, kH, dT, dW, dH, pT, pW, pH, ceilMode);
}
static inline void VolumetricMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, bool ceilMode) {
    return infer_type(input).VolumetricMaxPooling_updateGradInput(input, gradOutput, gradInput, indices, kT, kW, kH, dT, dW, dH, pT, pW, pH, ceilMode);
}
static inline void VolumetricDilatedMaxPooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) {
    return infer_type(input).VolumetricDilatedMaxPooling_updateOutput(input, output, indices, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, ceilMode);
}
static inline void VolumetricDilatedMaxPooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int kT, int kW, int kH, int dT, int dW, int dH, int pT, int pW, int pH, int dilationT, int dilationW, int dilationH, bool ceilMode) {
    return infer_type(input).VolumetricDilatedMaxPooling_updateGradInput(input, gradOutput, gradInput, indices, kT, kW, kH, dT, dW, dH, pT, pW, pH, dilationT, dilationW, dilationH, ceilMode);
}
static inline void VolumetricMaxUnpooling_updateOutput(const Tensor & input, const Tensor & output, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricMaxUnpooling_updateOutput(input, output, indices, oT, oW, oH, dT, dW, dH, pT, pW, pH);
}
static inline void VolumetricMaxUnpooling_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & indices, int oT, int oW, int oH, int dT, int dW, int dH, int pT, int pW, int pH) {
    return infer_type(input).VolumetricMaxUnpooling_updateGradInput(input, gradOutput, gradInput, indices, oT, oW, oH, dT, dW, dH, pT, pW, pH);
}
static inline void SpatialReflectionPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) {
    return infer_type(input).SpatialReflectionPadding_updateOutput(input, output, pad_l, pad_r, pad_t, pad_b);
}
static inline void SpatialReflectionPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) {
    return infer_type(input).SpatialReflectionPadding_updateGradInput(input, gradOutput, gradInput, pad_l, pad_r, pad_t, pad_b);
}
static inline void SpatialReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pad_l, int pad_r, int pad_t, int pad_b) {
    return infer_type(input).SpatialReplicationPadding_updateOutput(input, output, pad_l, pad_r, pad_t, pad_b);
}
static inline void SpatialReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pad_l, int pad_r, int pad_t, int pad_b) {
    return infer_type(input).SpatialReplicationPadding_updateGradInput(input, gradOutput, gradInput, pad_l, pad_r, pad_t, pad_b);
}
static inline void FeatureLPPooling_updateOutput(const Tensor & input, const Tensor & output, Scalar power, int width, int stride, bool batchMode) {
    return infer_type(input).FeatureLPPooling_updateOutput(input, output, power, width, stride, batchMode);
}
static inline void FeatureLPPooling_updateGradInput(const Tensor & gradOutput, const Tensor & input, const Tensor & output, const Tensor & gradInput, Scalar power, int width, int stride, bool batchMode) {
    return infer_type(gradOutput).FeatureLPPooling_updateGradInput(gradOutput, input, output, gradInput, power, width, stride, batchMode);
}
static inline void VolumetricReplicationPadding_updateOutput(const Tensor & input, const Tensor & output, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) {
    return infer_type(input).VolumetricReplicationPadding_updateOutput(input, output, pleft, pright, ptop, pbottom, pfront, pback);
}
static inline void VolumetricReplicationPadding_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int pleft, int pright, int ptop, int pbottom, int pfront, int pback) {
    return infer_type(input).VolumetricReplicationPadding_updateGradInput(input, gradOutput, gradInput, pleft, pright, ptop, pbottom, pfront, pback);
}
static inline void VolumetricUpSamplingNearest_updateOutput(const Tensor & input, const Tensor & output, int scale_factor) {
    return infer_type(input).VolumetricUpSamplingNearest_updateOutput(input, output, scale_factor);
}
static inline void VolumetricUpSamplingNearest_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, int scale_factor) {
    return infer_type(input).VolumetricUpSamplingNearest_updateGradInput(input, gradOutput, gradInput, scale_factor);
}
static inline void VolumetricUpSamplingTrilinear_updateOutput(const Tensor & input, const Tensor & output, int outputDepth, int outputHeight, int outputWidth) {
    return infer_type(input).VolumetricUpSamplingTrilinear_updateOutput(input, output, outputDepth, outputHeight, outputWidth);
}
static inline void VolumetricUpSamplingTrilinear_updateGradInput(const Tensor & gradOutput, const Tensor & gradInput, int nbatch, int nchannels, int inputDepth, int inputHeight, int inputWidth, int outputDepth, int outputHeight, int outputWidth) {
    return infer_type(gradOutput).VolumetricUpSamplingTrilinear_updateGradInput(gradOutput, gradInput, nbatch, nchannels, inputDepth, inputHeight, inputWidth, outputDepth, outputHeight, outputWidth);
}
static inline void SpatialCrossMapLRN_updateOutput(const Tensor & input, const Tensor & output, const Tensor & scale, int size, Scalar alpha, Scalar beta, Scalar k) {
    return infer_type(input).SpatialCrossMapLRN_updateOutput(input, output, scale, size, alpha, beta, k);
}
static inline void SpatialCrossMapLRN_updateGradInput(const Tensor & input, const Tensor & gradOutput, const Tensor & gradInput, const Tensor & scale, const Tensor & output, int size, Scalar alpha, Scalar beta, Scalar k) {
    return infer_type(input).SpatialCrossMapLRN_updateGradInput(input, gradOutput, gradInput, scale, output, size, alpha, beta, k);
}

}
