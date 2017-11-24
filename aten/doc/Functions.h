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

static inline Tensor & zeros_out(Tensor & result, IntList size);
static inline Tensor & zeros_like_out(Tensor & result, const Tensor & input);
static inline Tensor zeros_like(const Tensor & input);
static inline Tensor & ones_out(Tensor & result, IntList size);
static inline Tensor & ones_like_out(Tensor & result, const Tensor & input);
static inline Tensor ones_like(const Tensor & input);
static inline int64_t numel(const Tensor & self);
static inline Tensor & masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask);
static inline Tensor masked_select(const Tensor & self, const Tensor & mask);
static inline Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1);
static inline Tensor t(const Tensor & self);
static inline Tensor & squeeze_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor squeeze(const Tensor & self, int64_t dim);
static inline Tensor & squeeze_out(Tensor & result, const Tensor & self);
static inline Tensor squeeze(const Tensor & self);
static inline Tensor & unsqueeze_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor unsqueeze(const Tensor & self, int64_t dim);
static inline Tensor & nonzero_out(Tensor & result, const Tensor & self);
static inline Tensor nonzero(const Tensor & self);
static inline Tensor & index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
static inline Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
static inline Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor gather(const Tensor & self, int64_t dim, const Tensor & index);
static inline bool equal(const Tensor & self, const Tensor & other);
static inline Tensor & __and___out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor __and__(const Tensor & self, Scalar other);
static inline Tensor & __and___out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor __and__(const Tensor & self, const Tensor & other);
static inline Tensor & __iand__(Tensor & self, Scalar other);
static inline Tensor & __iand__(Tensor & self, const Tensor & other);
static inline Tensor & __or___out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor __or__(const Tensor & self, Scalar other);
static inline Tensor & __or___out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor __or__(const Tensor & self, const Tensor & other);
static inline Tensor & __ior__(Tensor & self, Scalar other);
static inline Tensor & __ior__(Tensor & self, const Tensor & other);
static inline Tensor & __xor___out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor __xor__(const Tensor & self, Scalar other);
static inline Tensor & __xor___out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor __xor__(const Tensor & self, const Tensor & other);
static inline Tensor & __ixor__(Tensor & self, Scalar other);
static inline Tensor & __ixor__(Tensor & self, const Tensor & other);
static inline Tensor & __lshift___out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor __lshift__(const Tensor & self, Scalar other);
static inline Tensor & __lshift___out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor __lshift__(const Tensor & self, const Tensor & other);
static inline Tensor & __ilshift__(Tensor & self, Scalar other);
static inline Tensor & __ilshift__(Tensor & self, const Tensor & other);
static inline Tensor & __rshift___out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor __rshift__(const Tensor & self, Scalar other);
static inline Tensor & __rshift___out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor __rshift__(const Tensor & self, const Tensor & other);
static inline Tensor & __irshift__(Tensor & self, Scalar other);
static inline Tensor & __irshift__(Tensor & self, const Tensor & other);
static inline Tensor & lt_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor lt(const Tensor & self, Scalar other);
static inline Tensor & lt_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor lt(const Tensor & self, const Tensor & other);
static inline Tensor & gt_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor gt(const Tensor & self, Scalar other);
static inline Tensor & gt_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor gt(const Tensor & self, const Tensor & other);
static inline Tensor & le_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor le(const Tensor & self, Scalar other);
static inline Tensor & le_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor le(const Tensor & self, const Tensor & other);
static inline Tensor & ge_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor ge(const Tensor & self, Scalar other);
static inline Tensor & ge_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor ge(const Tensor & self, const Tensor & other);
static inline Tensor & eq_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor eq(const Tensor & self, Scalar other);
static inline Tensor & eq_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor eq(const Tensor & self, const Tensor & other);
static inline Tensor & ne_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor ne(const Tensor & self, Scalar other);
static inline Tensor & ne_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor ne(const Tensor & self, const Tensor & other);
static inline std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor & min_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor min(const Tensor & self, const Tensor & other);
static inline Scalar min(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor & max_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor max(const Tensor & self, const Tensor & other);
static inline Scalar max(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Scalar median(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool descending=false);
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim=-1, bool descending=false);
static inline std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true);
static inline Tensor & abs_out(Tensor & destination, const Tensor & self);
static inline Tensor abs(const Tensor & self);
static inline Tensor & sigmoid_out(Tensor & result, const Tensor & self);
static inline Tensor sigmoid(const Tensor & self);
static inline Tensor & log_out(Tensor & result, const Tensor & self);
static inline Tensor log(const Tensor & self);
static inline Tensor & log1p_out(Tensor & result, const Tensor & self);
static inline Tensor log1p(const Tensor & self);
static inline Tensor & lgamma_out(Tensor & result, const Tensor & self);
static inline Tensor lgamma(const Tensor & self);
static inline Tensor & exp_out(Tensor & result, const Tensor & self);
static inline Tensor exp(const Tensor & self);
static inline Tensor & cos_out(Tensor & result, const Tensor & self);
static inline Tensor cos(const Tensor & self);
static inline Tensor & acos_out(Tensor & result, const Tensor & self);
static inline Tensor acos(const Tensor & self);
static inline Tensor & cosh_out(Tensor & result, const Tensor & self);
static inline Tensor cosh(const Tensor & self);
static inline Tensor & sin_out(Tensor & result, const Tensor & self);
static inline Tensor sin(const Tensor & self);
static inline Tensor & asin_out(Tensor & result, const Tensor & self);
static inline Tensor asin(const Tensor & self);
static inline Tensor & sinh_out(Tensor & result, const Tensor & self);
static inline Tensor sinh(const Tensor & self);
static inline Tensor & tan_out(Tensor & result, const Tensor & self);
static inline Tensor tan(const Tensor & self);
static inline Tensor & atan_out(Tensor & result, const Tensor & self);
static inline Tensor atan(const Tensor & self);
static inline Tensor & tanh_out(Tensor & result, const Tensor & self);
static inline Tensor tanh(const Tensor & self);
static inline Tensor & erf_out(Tensor & result, const Tensor & self);
static inline Tensor erf(const Tensor & self);
static inline Tensor & erfinv_out(Tensor & result, const Tensor & self);
static inline Tensor erfinv(const Tensor & self);
static inline Tensor & sqrt_out(Tensor & result, const Tensor & self);
static inline Tensor sqrt(const Tensor & self);
static inline Tensor & rsqrt_out(Tensor & result, const Tensor & self);
static inline Tensor rsqrt(const Tensor & self);
static inline Tensor & ceil_out(Tensor & result, const Tensor & self);
static inline Tensor ceil(const Tensor & self);
static inline Tensor & floor_out(Tensor & result, const Tensor & self);
static inline Tensor floor(const Tensor & self);
static inline Tensor & round_out(Tensor & result, const Tensor & self);
static inline Tensor round(const Tensor & self);
static inline Tensor & trunc_out(Tensor & result, const Tensor & self);
static inline Tensor trunc(const Tensor & self);
static inline Tensor & frac_out(Tensor & result, const Tensor & self);
static inline Tensor frac(const Tensor & self);
static inline Tensor & mean_out(Tensor & destination, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Scalar mean(const Tensor & self);
static inline Tensor & var_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Scalar var(const Tensor & self, bool unbiased=true);
static inline Tensor & std_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Scalar std(const Tensor & self, bool unbiased=true);
static inline Tensor & norm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, bool keepdim=false);
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim=false);
static inline Scalar norm(const Tensor & self, Scalar p=2);
static inline Tensor & renorm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
static inline Scalar dist(const Tensor & self, const Tensor & other, Scalar p=2);
static inline Tensor & reciprocal_out(Tensor & destination, const Tensor & self);
static inline Tensor reciprocal(const Tensor & self);
static inline Tensor & neg_out(Tensor & destination, const Tensor & self);
static inline Tensor neg(const Tensor & self);
static inline Tensor & atan2_out(Tensor & destination, const Tensor & self, const Tensor & other);
static inline Tensor atan2(const Tensor & self, const Tensor & other);
static inline Tensor & pow_out(Tensor & destination, const Tensor & self, Scalar exponent);
static inline Tensor pow(const Tensor & self, Scalar exponent);
static inline Tensor & pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent);
static inline Tensor pow(const Tensor & self, const Tensor & exponent);
static inline Tensor & lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight);
static inline Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight);
static inline Tensor & linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
static inline Tensor & logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
static inline Tensor & histc_out(Tensor & destination, const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0);
static inline Tensor histc(const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0);
static inline Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor sum(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Scalar sum(const Tensor & self);
static inline Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Scalar prod(const Tensor & self);
static inline Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor cumsum(const Tensor & self, int64_t dim);
static inline Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor cumprod(const Tensor & self, int64_t dim);
static inline Tensor & sign_out(Tensor & result, const Tensor & self);
static inline Tensor sign(const Tensor & self);
static inline Scalar trace(const Tensor & self);
static inline Tensor & add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha=1);
static inline Tensor add(const Tensor & self, Scalar other, Scalar alpha=1);
static inline Tensor & add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1);
static inline Tensor add(const Tensor & self, const Tensor & other, Scalar alpha=1);
static inline Tensor & add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha=1);
static inline Tensor add(const Tensor & self, SparseTensor other, Scalar alpha=1);
static inline Tensor & sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha=1);
static inline Tensor sub(const Tensor & self, Scalar other, Scalar alpha=1);
static inline Tensor & sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha=1);
static inline Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha=1);
static inline Tensor & mul_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor mul(const Tensor & self, Scalar other);
static inline Tensor & mul_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor mul(const Tensor & self, const Tensor & other);
static inline Tensor & div_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor div(const Tensor & self, Scalar other);
static inline Tensor & div_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor div(const Tensor & self, const Tensor & other);
static inline Tensor & fmod_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor fmod(const Tensor & self, Scalar other);
static inline Tensor & fmod_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor fmod(const Tensor & self, const Tensor & other);
static inline Tensor & remainder_out(Tensor & result, const Tensor & self, Scalar other);
static inline Tensor remainder(const Tensor & self, Scalar other);
static inline Tensor & remainder_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor remainder(const Tensor & self, const Tensor & other);
static inline Tensor & clamp_out(Tensor & destination, const Tensor & self, Scalar min, Scalar max);
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max);
static inline Tensor & clamp_out(Tensor & result, const Tensor & self, Scalar min);
static inline Tensor clamp(const Tensor & self, Scalar min);
static inline Scalar dot(const Tensor & self, const Tensor & tensor);
static inline Tensor & tril_out(Tensor & destination, const Tensor & self, int64_t diagonal=0);
static inline Tensor tril(const Tensor & self, int64_t diagonal=0);
static inline Tensor & triu_out(Tensor & destination, const Tensor & self, int64_t diagonal=0);
static inline Tensor triu(const Tensor & self, int64_t diagonal=0);
static inline Tensor & cross_out(Tensor & destination, const Tensor & self, const Tensor & other, int64_t dim=-1);
static inline Tensor cross(const Tensor & self, const Tensor & other, int64_t dim=-1);
static inline Tensor & eye_out(Tensor & result, int64_t n, int64_t m=1);
static inline Tensor & diag_out(Tensor & result, const Tensor & self, int64_t diagonal=0);
static inline Tensor diag(const Tensor & self, int64_t diagonal=0);
static inline Tensor & addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2);
static inline Tensor ger(const Tensor & self, const Tensor & vec2);
static inline Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec);
static inline Tensor mv(const Tensor & self, const Tensor & vec);
static inline Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
static inline Tensor mm(const Tensor & self, const Tensor & mat2);
static inline Tensor & bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
static inline Tensor bmm(const Tensor & self, const Tensor & mat2);
static inline Tensor & addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
static inline Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
static inline Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
static inline Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
static inline Tensor & addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
static inline Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value=1);
static inline std::tuple<Tensor &,Tensor &> gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor &,Tensor &> gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A);
static inline std::tuple<Tensor &,Tensor &> trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false);
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper=true, bool transpose=false, bool unitriangular=false);
static inline std::tuple<Tensor &,Tensor &> symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors=false, bool upper=true);
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors=false, bool upper=true);
static inline std::tuple<Tensor &,Tensor &> eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors=false);
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors=false);
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some=true);
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some=true);
static inline Tensor & inverse_out(Tensor & output, const Tensor & self);
static inline Tensor inverse(const Tensor & self);
static inline Tensor & potrf_out(Tensor & output, const Tensor & self, bool upper=true);
static inline Tensor potrf(const Tensor & self, bool upper=true);
static inline Tensor & potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper=true);
static inline Tensor potrs(const Tensor & self, const Tensor & input2, bool upper=true);
static inline Tensor & potri_out(Tensor & output, const Tensor & self, bool upper=true);
static inline Tensor potri(const Tensor & self, bool upper=true);
static inline std::tuple<Tensor &,Tensor &> pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper=true, Scalar tol=-1);
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper=true, Scalar tol=-1);
static inline std::tuple<Tensor &,Tensor &> qr_out(Tensor & res1, Tensor & res2, const Tensor & self);
static inline std::tuple<Tensor,Tensor> qr(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self);
static inline std::tuple<Tensor,Tensor> geqrf(const Tensor & self);
static inline Tensor & orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2);
static inline Tensor orgqr(const Tensor & self, const Tensor & input2);
static inline Tensor & ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false);
static inline Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left=true, bool transpose=false);
static inline std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, const Tensor & info={}, bool pivot=true);
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self, const Tensor & info={}, bool pivot=true);
static inline Tensor & btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots);
static inline Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots);
static inline Tensor & randperm_out(Tensor & result, int64_t n, Generator * generator=nullptr);
static inline Tensor & multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr);
static inline Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, const Tensor & means, double std=1, Generator * generator=nullptr);
static inline Tensor normal(const Tensor & means, double std=1, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor normal(double mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, const Tensor & means, const Tensor & std, Generator * generator=nullptr);
static inline Tensor normal(const Tensor & means, const Tensor & std, Generator * generator=nullptr);
static inline Tensor & rand_out(Tensor & result, IntList size, Generator * generator=nullptr);
static inline Tensor & randn_out(Tensor & result, IntList size, Generator * generator=nullptr);
static inline Tensor & select_out(Tensor & result, const Tensor & self, int64_t dim, int64_t sliceIndex);
static inline Tensor select(const Tensor & self, int64_t dim, int64_t sliceIndex);
static inline Tensor & _unnarrow_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize);
static inline Tensor _unnarrow(const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize);
static inline Tensor & cat_out(Tensor & self, TensorList tensors, int64_t dim);
static inline Tensor cat(TensorList tensors, int64_t dim);
static inline Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true);
static inline Tensor binary_cross_entropy(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true);
static inline Tensor & binary_cross_entropy_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average);
static inline Tensor binary_cross_entropy_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average);
static inline Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average);
static inline Tensor binary_cross_entropy_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average);
static inline Tensor & kl_div_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor kl_div(const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor & kl_div_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor kl_div_forward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & kl_div_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor kl_div_backward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor l1_loss(const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor & l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & mse_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor mse_loss(const Tensor & input, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor & mse_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, bool reduce);
static inline Tensor mse_loss_forward(const Tensor & input, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce);
static inline Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & multi_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true);
static inline Tensor multi_margin_loss(const Tensor & input, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true);
static inline Tensor & multi_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor multi_margin_loss_forward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor multi_margin_loss_backward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor multilabel_margin_loss(const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor & multilabel_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target);
static inline Tensor multilabel_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target);
static inline Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target);
static inline Tensor multilabel_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target);
static inline Tensor & nll_loss_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100);
static inline Tensor nll_loss(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100);
static inline Tensor & nll_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor nll_loss_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor nll_loss_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor & nll_loss2d_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100);
static inline Tensor nll_loss2d(const Tensor & input, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100);
static inline Tensor & nll_loss2d_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor nll_loss2d_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor nll_loss2d_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight);
static inline Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor smooth_l1_loss(const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor & smooth_l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor smooth_l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor smooth_l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & soft_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor soft_margin_loss(const Tensor & input, const Tensor & target, bool size_average=true);
static inline Tensor & soft_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor soft_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor soft_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average);
static inline Tensor & elu_out(Tensor & output, const Tensor & input, Scalar alpha=1, bool inplace=false);
static inline Tensor elu(const Tensor & input, Scalar alpha=1, bool inplace=false);
static inline Tensor & elu_forward_out(Tensor & output, const Tensor & input, Scalar alpha, bool inplace);
static inline Tensor elu_forward(const Tensor & input, Scalar alpha, bool inplace);
static inline Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output);
static inline Tensor elu_backward(const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output);
static inline Tensor & glu_out(Tensor & output, const Tensor & input, int64_t dim=-1);
static inline Tensor glu(const Tensor & input, int64_t dim=-1);
static inline Tensor & glu_forward_out(Tensor & output, const Tensor & input, int64_t dim);
static inline Tensor glu_forward(const Tensor & input, int64_t dim);
static inline Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, int64_t dim);
static inline Tensor glu_backward(const Tensor & grad_output, const Tensor & input, int64_t dim);
static inline Tensor & hardshrink_out(Tensor & output, const Tensor & input, Scalar lambd=0.5);
static inline Tensor hardshrink(const Tensor & input, Scalar lambd=0.5);
static inline Tensor & hardshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd);
static inline Tensor hardshrink_forward(const Tensor & input, Scalar lambd);
static inline Tensor & hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd);
static inline Tensor hardshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd);
static inline Tensor & hardtanh_out(Tensor & output, const Tensor & input, Scalar min_val=-1, Scalar max_val=1, bool inplace=false);
static inline Tensor hardtanh(const Tensor & input, Scalar min_val=-1, Scalar max_val=1, bool inplace=false);
static inline Tensor & hardtanh_forward_out(Tensor & output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace);
static inline Tensor hardtanh_forward(const Tensor & input, Scalar min_val, Scalar max_val, bool inplace);
static inline Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace);
static inline Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace);
static inline Tensor & leaky_relu_out(Tensor & output, const Tensor & input, Scalar negative_slope=0.01, bool inplace=false);
static inline Tensor leaky_relu(const Tensor & input, Scalar negative_slope=0.01, bool inplace=false);
static inline Tensor & leaky_relu_forward_out(Tensor & output, const Tensor & input, Scalar negative_slope, bool inplace);
static inline Tensor leaky_relu_forward(const Tensor & input, Scalar negative_slope, bool inplace);
static inline Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace);
static inline Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace);
static inline Tensor & log_sigmoid_out(Tensor & output, const Tensor & input);
static inline Tensor log_sigmoid(const Tensor & input);
static inline Tensor & log_sigmoid_forward_out(Tensor & output, const Tensor & input, const Tensor & buffer);
static inline Tensor log_sigmoid_forward(const Tensor & input, const Tensor & buffer);
static inline Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & buffer);
static inline Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & input, const Tensor & buffer);
static inline Tensor & log_softmax_out(Tensor & output, const Tensor & input, int64_t dim);
static inline Tensor log_softmax(const Tensor & input, int64_t dim);
static inline Tensor & log_softmax_forward_out(Tensor & output, const Tensor & input);
static inline Tensor log_softmax_forward(const Tensor & input);
static inline Tensor & log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output);
static inline Tensor log_softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output);
static inline Tensor & prelu_out(Tensor & output, const Tensor & input, const Tensor & weight);
static inline Tensor prelu(const Tensor & input, const Tensor & weight);
static inline Tensor & prelu_forward_out(Tensor & output, const Tensor & input, const Tensor & weight);
static inline Tensor prelu_forward(const Tensor & input, const Tensor & weight);
static inline std::tuple<Tensor &,Tensor &> prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & input, const Tensor & weight);
static inline std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, std::array<bool, 2> output_mask={true, true});
static inline Tensor & rrelu_out(Tensor & output, const Tensor & input, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=false, bool inplace=false, Generator * generator=nullptr);
static inline Tensor rrelu(const Tensor & input, Scalar lower=0.125, Scalar upper=0.333333333333, bool training=false, bool inplace=false, Generator * generator=nullptr);
static inline Tensor & rrelu_forward_out(Tensor & output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise);
static inline Tensor rrelu_forward(const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise);
static inline Tensor & rrelu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise);
static inline Tensor rrelu_backward(const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise);
static inline Tensor & softmax_out(Tensor & output, const Tensor & input, int64_t dim);
static inline Tensor softmax(const Tensor & input, int64_t dim);
static inline Tensor & softmax_forward_out(Tensor & output, const Tensor & input);
static inline Tensor softmax_forward(const Tensor & input);
static inline Tensor & softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output);
static inline Tensor softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output);
static inline Tensor & softplus_out(Tensor & output, const Tensor & input, Scalar beta=1, Scalar threshold=20);
static inline Tensor softplus(const Tensor & input, Scalar beta=1, Scalar threshold=20);
static inline Tensor & softplus_forward_out(Tensor & output, const Tensor & input, Scalar beta, Scalar threshold);
static inline Tensor softplus_forward(const Tensor & input, Scalar beta, Scalar threshold);
static inline Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output);
static inline Tensor softplus_backward(const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output);
static inline Tensor & softshrink_out(Tensor & output, const Tensor & input, Scalar lambd=0.5);
static inline Tensor softshrink(const Tensor & input, Scalar lambd=0.5);
static inline Tensor & softshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd);
static inline Tensor softshrink_forward(const Tensor & input, Scalar lambd);
static inline Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd);
static inline Tensor softshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd);
static inline Tensor & threshold_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace=false);
static inline Tensor threshold(const Tensor & input, Scalar threshold, Scalar value, bool inplace=false);
static inline Tensor & threshold_forward_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace);
static inline Tensor threshold_forward(const Tensor & input, Scalar threshold, Scalar value, bool inplace);
static inline Tensor & threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace);
static inline Tensor threshold_backward(const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & input, IntList output_size);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d_forward(const Tensor & input, IntList output_size);
static inline Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices);
static inline Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices);
static inline Tensor & avg_pool2d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor avg_pool2d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor & avg_pool2d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool3d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor avg_pool3d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor & avg_pool3d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline std::tuple<Tensor &,Tensor &> max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor,Tensor> max_pool2d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor &,Tensor &> max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline std::tuple<Tensor,Tensor> max_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline Tensor & max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor max_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline std::tuple<Tensor &,Tensor &> max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor,Tensor> max_pool3d(const Tensor & input, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor &,Tensor &> max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline std::tuple<Tensor,Tensor> max_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline Tensor & max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor max_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor & max_unpool2d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d(const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool2d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d_forward(const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool3d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & max_unpool3d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d_forward(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & _sigmoid_out(Tensor & output, const Tensor & input);
static inline Tensor _sigmoid(const Tensor & input);
static inline Tensor & _sigmoid_forward_out(Tensor & output, const Tensor & input);
static inline Tensor _sigmoid_forward(const Tensor & input);
static inline Tensor & _sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
static inline Tensor _sigmoid_backward(const Tensor & grad_output, const Tensor & output);
static inline Tensor & _tanh_out(Tensor & output, const Tensor & input);
static inline Tensor _tanh(const Tensor & input);
static inline Tensor & _tanh_forward_out(Tensor & output, const Tensor & input);
static inline Tensor _tanh_forward(const Tensor & input);
static inline Tensor & _tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
static inline Tensor _tanh_backward(const Tensor & grad_output, const Tensor & output);
static inline Tensor & batch_norm_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline Tensor & batch_norm_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std);
static inline Tensor batch_norm_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std);
static inline std::tuple<Tensor &,Tensor &,Tensor &> batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std);
static inline std::tuple<Tensor,Tensor,Tensor> batch_norm_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv_transpose2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor & conv_transpose2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline Tensor conv_transpose2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> conv_transpose2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv_transpose3d_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor & conv_transpose3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input);
static inline Tensor conv_transpose3d_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> conv_transpose3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor conv2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor & conv2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline Tensor conv2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> conv2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor conv3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor & conv3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput);
static inline Tensor conv3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> conv3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv_dilated2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor conv_dilated2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor & conv_dilated2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline Tensor conv_dilated2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> conv_dilated2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true});
static inline Tensor & conv_dilated3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor conv_dilated3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor & conv_dilated3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline Tensor conv_dilated3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> conv_dilated3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask={true, true, true});
static inline std::vector<Tensor> split(Tensor self, int64_t split_size, int64_t dim=0);
static inline std::vector<Tensor> chunk(Tensor self, int64_t chunks, int64_t dim=0);

static inline Type & infer_type(const Tensor & t) {
  AT_ASSERT(t.defined(), "undefined Tensor");
  return t.type();
}
static inline Type & infer_type(const TensorList & tl) {
  AT_ASSERT(tl.size() > 0, "expected a non-empty list of Tensors");
  return tl[0].type();
}
// function definitions are all static inline because
// they are one-line statically dispatched functions that
// invoke the actual dynamic dispatch on the correct argument
static inline Tensor & zeros_out(Tensor & result, IntList size) {
    return infer_type(result).zeros_out(result, size);
}
static inline Tensor & zeros_like_out(Tensor & result, const Tensor & input) {
    return infer_type(result).zeros_like_out(result, input);
}
static inline Tensor zeros_like(const Tensor & input) {
    return infer_type(input).zeros_like(input);
}
static inline Tensor & ones_out(Tensor & result, IntList size) {
    return infer_type(result).ones_out(result, size);
}
static inline Tensor & ones_like_out(Tensor & result, const Tensor & input) {
    return infer_type(result).ones_like_out(result, input);
}
static inline Tensor ones_like(const Tensor & input) {
    return infer_type(input).ones_like(input);
}
static inline int64_t numel(const Tensor & self) {
    return infer_type(self).numel(self);
}
static inline Tensor & masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask) {
    return infer_type(self).masked_select_out(result, self, mask);
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
static inline Tensor & squeeze_out(Tensor & result, const Tensor & self, int64_t dim) {
    return infer_type(self).squeeze_out(result, self, dim);
}
static inline Tensor squeeze(const Tensor & self, int64_t dim) {
    return infer_type(self).squeeze(self, dim);
}
static inline Tensor & squeeze_out(Tensor & result, const Tensor & self) {
    return infer_type(self).squeeze_out(result, self);
}
static inline Tensor squeeze(const Tensor & self) {
    return infer_type(self).squeeze(self);
}
static inline Tensor & unsqueeze_out(Tensor & result, const Tensor & self, int64_t dim) {
    return infer_type(self).unsqueeze_out(result, self, dim);
}
static inline Tensor unsqueeze(const Tensor & self, int64_t dim) {
    return infer_type(self).unsqueeze(self, dim);
}
static inline Tensor & nonzero_out(Tensor & result, const Tensor & self) {
    return infer_type(self).nonzero_out(result, self);
}
static inline Tensor nonzero(const Tensor & self) {
    return infer_type(self).nonzero(self);
}
static inline Tensor & index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).index_select_out(result, self, dim, index);
}
static inline Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).index_select(self, dim, index);
}
static inline Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step) {
    return infer_type(result).range_out(result, start, end, step);
}
static inline Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) {
    return infer_type(result).arange_out(result, start, end, step);
}
static inline Tensor & gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).gather_out(result, self, dim, index);
}
static inline Tensor gather(const Tensor & self, int64_t dim, const Tensor & index) {
    return infer_type(self).gather(self, dim, index);
}
static inline bool equal(const Tensor & self, const Tensor & other) {
    return infer_type(self).equal(self, other);
}
static inline Tensor & __and___out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).__and___out(result, self, other);
}
static inline Tensor __and__(const Tensor & self, Scalar other) {
    return infer_type(self).__and__(self, other);
}
static inline Tensor & __and___out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).__and___out(result, self, other);
}
static inline Tensor __and__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__and__(self, other);
}
static inline Tensor & __iand__(Tensor & self, Scalar other) {
    return infer_type(self).__iand__(self, other);
}
static inline Tensor & __iand__(Tensor & self, const Tensor & other) {
    return infer_type(self).__iand__(self, other);
}
static inline Tensor & __or___out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).__or___out(result, self, other);
}
static inline Tensor __or__(const Tensor & self, Scalar other) {
    return infer_type(self).__or__(self, other);
}
static inline Tensor & __or___out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).__or___out(result, self, other);
}
static inline Tensor __or__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__or__(self, other);
}
static inline Tensor & __ior__(Tensor & self, Scalar other) {
    return infer_type(self).__ior__(self, other);
}
static inline Tensor & __ior__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ior__(self, other);
}
static inline Tensor & __xor___out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).__xor___out(result, self, other);
}
static inline Tensor __xor__(const Tensor & self, Scalar other) {
    return infer_type(self).__xor__(self, other);
}
static inline Tensor & __xor___out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).__xor___out(result, self, other);
}
static inline Tensor __xor__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__xor__(self, other);
}
static inline Tensor & __ixor__(Tensor & self, Scalar other) {
    return infer_type(self).__ixor__(self, other);
}
static inline Tensor & __ixor__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ixor__(self, other);
}
static inline Tensor & __lshift___out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).__lshift___out(result, self, other);
}
static inline Tensor __lshift__(const Tensor & self, Scalar other) {
    return infer_type(self).__lshift__(self, other);
}
static inline Tensor & __lshift___out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).__lshift___out(result, self, other);
}
static inline Tensor __lshift__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__lshift__(self, other);
}
static inline Tensor & __ilshift__(Tensor & self, Scalar other) {
    return infer_type(self).__ilshift__(self, other);
}
static inline Tensor & __ilshift__(Tensor & self, const Tensor & other) {
    return infer_type(self).__ilshift__(self, other);
}
static inline Tensor & __rshift___out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).__rshift___out(result, self, other);
}
static inline Tensor __rshift__(const Tensor & self, Scalar other) {
    return infer_type(self).__rshift__(self, other);
}
static inline Tensor & __rshift___out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).__rshift___out(result, self, other);
}
static inline Tensor __rshift__(const Tensor & self, const Tensor & other) {
    return infer_type(self).__rshift__(self, other);
}
static inline Tensor & __irshift__(Tensor & self, Scalar other) {
    return infer_type(self).__irshift__(self, other);
}
static inline Tensor & __irshift__(Tensor & self, const Tensor & other) {
    return infer_type(self).__irshift__(self, other);
}
static inline Tensor & lt_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).lt_out(result, self, other);
}
static inline Tensor lt(const Tensor & self, Scalar other) {
    return infer_type(self).lt(self, other);
}
static inline Tensor & lt_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).lt_out(result, self, other);
}
static inline Tensor lt(const Tensor & self, const Tensor & other) {
    return infer_type(self).lt(self, other);
}
static inline Tensor & gt_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).gt_out(result, self, other);
}
static inline Tensor gt(const Tensor & self, Scalar other) {
    return infer_type(self).gt(self, other);
}
static inline Tensor & gt_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).gt_out(result, self, other);
}
static inline Tensor gt(const Tensor & self, const Tensor & other) {
    return infer_type(self).gt(self, other);
}
static inline Tensor & le_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).le_out(result, self, other);
}
static inline Tensor le(const Tensor & self, Scalar other) {
    return infer_type(self).le(self, other);
}
static inline Tensor & le_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).le_out(result, self, other);
}
static inline Tensor le(const Tensor & self, const Tensor & other) {
    return infer_type(self).le(self, other);
}
static inline Tensor & ge_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).ge_out(result, self, other);
}
static inline Tensor ge(const Tensor & self, Scalar other) {
    return infer_type(self).ge(self, other);
}
static inline Tensor & ge_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).ge_out(result, self, other);
}
static inline Tensor ge(const Tensor & self, const Tensor & other) {
    return infer_type(self).ge(self, other);
}
static inline Tensor & eq_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).eq_out(result, self, other);
}
static inline Tensor eq(const Tensor & self, Scalar other) {
    return infer_type(self).eq(self, other);
}
static inline Tensor & eq_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).eq_out(result, self, other);
}
static inline Tensor eq(const Tensor & self, const Tensor & other) {
    return infer_type(self).eq(self, other);
}
static inline Tensor & ne_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).ne_out(result, self, other);
}
static inline Tensor ne(const Tensor & self, Scalar other) {
    return infer_type(self).ne(self, other);
}
static inline Tensor & ne_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).ne_out(result, self, other);
}
static inline Tensor ne(const Tensor & self, const Tensor & other) {
    return infer_type(self).ne(self, other);
}
static inline std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).min_out(min, min_indices, self, dim, keepdim);
}
static inline std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).min(self, dim, keepdim);
}
static inline Tensor & min_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).min_out(result, self, other);
}
static inline Tensor min(const Tensor & self, const Tensor & other) {
    return infer_type(self).min(self, other);
}
static inline Scalar min(const Tensor & self) {
    return infer_type(self).min(self);
}
static inline std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).max_out(max, max_indices, self, dim, keepdim);
}
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).max(self, dim, keepdim);
}
static inline Tensor & max_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).max_out(result, self, other);
}
static inline Tensor max(const Tensor & self, const Tensor & other) {
    return infer_type(self).max(self, other);
}
static inline Scalar max(const Tensor & self) {
    return infer_type(self).max(self);
}
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    return infer_type(self).kthvalue_out(values, indices, self, k, dim, keepdim);
}
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim) {
    return infer_type(self).kthvalue(self, k, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mode_out(values, indices, self, dim, keepdim);
}
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mode(self, dim, keepdim);
}
static inline std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).median_out(values, indices, self, dim, keepdim);
}
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).median(self, dim, keepdim);
}
static inline Scalar median(const Tensor & self) {
    return infer_type(self).median(self);
}
static inline std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending) {
    return infer_type(self).sort_out(values, indices, self, dim, descending);
}
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending) {
    return infer_type(self).sort(self, dim, descending);
}
static inline std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    return infer_type(self).topk_out(values, indices, self, k, dim, largest, sorted);
}
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted) {
    return infer_type(self).topk(self, k, dim, largest, sorted);
}
static inline Tensor & abs_out(Tensor & destination, const Tensor & self) {
    return infer_type(self).abs_out(destination, self);
}
static inline Tensor abs(const Tensor & self) {
    return infer_type(self).abs(self);
}
static inline Tensor & sigmoid_out(Tensor & result, const Tensor & self) {
    return infer_type(self).sigmoid_out(result, self);
}
static inline Tensor sigmoid(const Tensor & self) {
    return infer_type(self).sigmoid(self);
}
static inline Tensor & log_out(Tensor & result, const Tensor & self) {
    return infer_type(self).log_out(result, self);
}
static inline Tensor log(const Tensor & self) {
    return infer_type(self).log(self);
}
static inline Tensor & log1p_out(Tensor & result, const Tensor & self) {
    return infer_type(self).log1p_out(result, self);
}
static inline Tensor log1p(const Tensor & self) {
    return infer_type(self).log1p(self);
}
static inline Tensor & lgamma_out(Tensor & result, const Tensor & self) {
    return infer_type(self).lgamma_out(result, self);
}
static inline Tensor lgamma(const Tensor & self) {
    return infer_type(self).lgamma(self);
}
static inline Tensor & exp_out(Tensor & result, const Tensor & self) {
    return infer_type(self).exp_out(result, self);
}
static inline Tensor exp(const Tensor & self) {
    return infer_type(self).exp(self);
}
static inline Tensor & cos_out(Tensor & result, const Tensor & self) {
    return infer_type(self).cos_out(result, self);
}
static inline Tensor cos(const Tensor & self) {
    return infer_type(self).cos(self);
}
static inline Tensor & acos_out(Tensor & result, const Tensor & self) {
    return infer_type(self).acos_out(result, self);
}
static inline Tensor acos(const Tensor & self) {
    return infer_type(self).acos(self);
}
static inline Tensor & cosh_out(Tensor & result, const Tensor & self) {
    return infer_type(self).cosh_out(result, self);
}
static inline Tensor cosh(const Tensor & self) {
    return infer_type(self).cosh(self);
}
static inline Tensor & sin_out(Tensor & result, const Tensor & self) {
    return infer_type(self).sin_out(result, self);
}
static inline Tensor sin(const Tensor & self) {
    return infer_type(self).sin(self);
}
static inline Tensor & asin_out(Tensor & result, const Tensor & self) {
    return infer_type(self).asin_out(result, self);
}
static inline Tensor asin(const Tensor & self) {
    return infer_type(self).asin(self);
}
static inline Tensor & sinh_out(Tensor & result, const Tensor & self) {
    return infer_type(self).sinh_out(result, self);
}
static inline Tensor sinh(const Tensor & self) {
    return infer_type(self).sinh(self);
}
static inline Tensor & tan_out(Tensor & result, const Tensor & self) {
    return infer_type(self).tan_out(result, self);
}
static inline Tensor tan(const Tensor & self) {
    return infer_type(self).tan(self);
}
static inline Tensor & atan_out(Tensor & result, const Tensor & self) {
    return infer_type(self).atan_out(result, self);
}
static inline Tensor atan(const Tensor & self) {
    return infer_type(self).atan(self);
}
static inline Tensor & tanh_out(Tensor & result, const Tensor & self) {
    return infer_type(self).tanh_out(result, self);
}
static inline Tensor tanh(const Tensor & self) {
    return infer_type(self).tanh(self);
}
static inline Tensor & erf_out(Tensor & result, const Tensor & self) {
    return infer_type(self).erf_out(result, self);
}
static inline Tensor erf(const Tensor & self) {
    return infer_type(self).erf(self);
}
static inline Tensor & erfinv_out(Tensor & result, const Tensor & self) {
    return infer_type(self).erfinv_out(result, self);
}
static inline Tensor erfinv(const Tensor & self) {
    return infer_type(self).erfinv(self);
}
static inline Tensor & sqrt_out(Tensor & result, const Tensor & self) {
    return infer_type(self).sqrt_out(result, self);
}
static inline Tensor sqrt(const Tensor & self) {
    return infer_type(self).sqrt(self);
}
static inline Tensor & rsqrt_out(Tensor & result, const Tensor & self) {
    return infer_type(self).rsqrt_out(result, self);
}
static inline Tensor rsqrt(const Tensor & self) {
    return infer_type(self).rsqrt(self);
}
static inline Tensor & ceil_out(Tensor & result, const Tensor & self) {
    return infer_type(self).ceil_out(result, self);
}
static inline Tensor ceil(const Tensor & self) {
    return infer_type(self).ceil(self);
}
static inline Tensor & floor_out(Tensor & result, const Tensor & self) {
    return infer_type(self).floor_out(result, self);
}
static inline Tensor floor(const Tensor & self) {
    return infer_type(self).floor(self);
}
static inline Tensor & round_out(Tensor & result, const Tensor & self) {
    return infer_type(self).round_out(result, self);
}
static inline Tensor round(const Tensor & self) {
    return infer_type(self).round(self);
}
static inline Tensor & trunc_out(Tensor & result, const Tensor & self) {
    return infer_type(self).trunc_out(result, self);
}
static inline Tensor trunc(const Tensor & self) {
    return infer_type(self).trunc(self);
}
static inline Tensor & frac_out(Tensor & result, const Tensor & self) {
    return infer_type(self).frac_out(result, self);
}
static inline Tensor frac(const Tensor & self) {
    return infer_type(self).frac(self);
}
static inline Tensor & mean_out(Tensor & destination, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mean_out(destination, self, dim, keepdim);
}
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mean(self, dim, keepdim);
}
static inline Scalar mean(const Tensor & self) {
    return infer_type(self).mean(self);
}
static inline Tensor & var_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).var_out(destination, self, dim, unbiased, keepdim);
}
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).var(self, dim, unbiased, keepdim);
}
static inline Scalar var(const Tensor & self, bool unbiased) {
    return infer_type(self).var(self, unbiased);
}
static inline Tensor & std_out(Tensor & destination, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).std_out(destination, self, dim, unbiased, keepdim);
}
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).std(self, dim, unbiased, keepdim);
}
static inline Scalar std(const Tensor & self, bool unbiased) {
    return infer_type(self).std(self, unbiased);
}
static inline Tensor & norm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    return infer_type(self).norm_out(destination, self, p, dim, keepdim);
}
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    return infer_type(self).norm(self, p, dim, keepdim);
}
static inline Scalar norm(const Tensor & self, Scalar p) {
    return infer_type(self).norm(self, p);
}
static inline Tensor & renorm_out(Tensor & destination, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    return infer_type(self).renorm_out(destination, self, p, dim, maxnorm);
}
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    return infer_type(self).renorm(self, p, dim, maxnorm);
}
static inline Scalar dist(const Tensor & self, const Tensor & other, Scalar p) {
    return infer_type(self).dist(self, other, p);
}
static inline Tensor & reciprocal_out(Tensor & destination, const Tensor & self) {
    return infer_type(self).reciprocal_out(destination, self);
}
static inline Tensor reciprocal(const Tensor & self) {
    return infer_type(self).reciprocal(self);
}
static inline Tensor & neg_out(Tensor & destination, const Tensor & self) {
    return infer_type(self).neg_out(destination, self);
}
static inline Tensor neg(const Tensor & self) {
    return infer_type(self).neg(self);
}
static inline Tensor & atan2_out(Tensor & destination, const Tensor & self, const Tensor & other) {
    return infer_type(self).atan2_out(destination, self, other);
}
static inline Tensor atan2(const Tensor & self, const Tensor & other) {
    return infer_type(self).atan2(self, other);
}
static inline Tensor & pow_out(Tensor & destination, const Tensor & self, Scalar exponent) {
    return infer_type(self).pow_out(destination, self, exponent);
}
static inline Tensor pow(const Tensor & self, Scalar exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & pow_out(Tensor & destination, const Tensor & self, const Tensor & exponent) {
    return infer_type(self).pow_out(destination, self, exponent);
}
static inline Tensor pow(const Tensor & self, const Tensor & exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & lerp_out(Tensor & destination, const Tensor & self, const Tensor & end, Scalar weight) {
    return infer_type(self).lerp_out(destination, self, end, weight);
}
static inline Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight) {
    return infer_type(self).lerp(self, end, weight);
}
static inline Tensor & linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) {
    return infer_type(result).linspace_out(result, start, end, steps);
}
static inline Tensor & logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps) {
    return infer_type(result).logspace_out(result, start, end, steps);
}
static inline Tensor & histc_out(Tensor & destination, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    return infer_type(self).histc_out(destination, self, bins, min, max);
}
static inline Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    return infer_type(self).histc(self, bins, min, max);
}
static inline Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).sum_out(result, self, dim, keepdim);
}
static inline Tensor sum(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).sum(self, dim, keepdim);
}
static inline Scalar sum(const Tensor & self) {
    return infer_type(self).sum(self);
}
static inline Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).prod_out(result, self, dim, keepdim);
}
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).prod(self, dim, keepdim);
}
static inline Scalar prod(const Tensor & self) {
    return infer_type(self).prod(self);
}
static inline Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim) {
    return infer_type(self).cumsum_out(result, self, dim);
}
static inline Tensor cumsum(const Tensor & self, int64_t dim) {
    return infer_type(self).cumsum(self, dim);
}
static inline Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim) {
    return infer_type(self).cumprod_out(result, self, dim);
}
static inline Tensor cumprod(const Tensor & self, int64_t dim) {
    return infer_type(self).cumprod(self, dim);
}
static inline Tensor & sign_out(Tensor & result, const Tensor & self) {
    return infer_type(self).sign_out(result, self);
}
static inline Tensor sign(const Tensor & self) {
    return infer_type(self).sign(self);
}
static inline Scalar trace(const Tensor & self) {
    return infer_type(self).trace(self);
}
static inline Tensor & add_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) {
    return infer_type(self).add_out(result, self, other, alpha);
}
static inline Tensor add(const Tensor & self, Scalar other, Scalar alpha) {
    return infer_type(self).add(self, other, alpha);
}
static inline Tensor & add_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) {
    return infer_type(self).add_out(result, self, other, alpha);
}
static inline Tensor add(const Tensor & self, const Tensor & other, Scalar alpha) {
    return infer_type(self).add(self, other, alpha);
}
static inline Tensor & add_out(Tensor & result, const Tensor & self, SparseTensor other, Scalar alpha) {
    return infer_type(self).add_out(result, self, other, alpha);
}
static inline Tensor add(const Tensor & self, SparseTensor other, Scalar alpha) {
    return infer_type(self).add(self, other, alpha);
}
static inline Tensor & sub_out(Tensor & result, const Tensor & self, Scalar other, Scalar alpha) {
    return infer_type(self).sub_out(result, self, other, alpha);
}
static inline Tensor sub(const Tensor & self, Scalar other, Scalar alpha) {
    return infer_type(self).sub(self, other, alpha);
}
static inline Tensor & sub_out(Tensor & result, const Tensor & self, const Tensor & other, Scalar alpha) {
    return infer_type(self).sub_out(result, self, other, alpha);
}
static inline Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha) {
    return infer_type(self).sub(self, other, alpha);
}
static inline Tensor & mul_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).mul_out(result, self, other);
}
static inline Tensor mul(const Tensor & self, Scalar other) {
    return infer_type(self).mul(self, other);
}
static inline Tensor & mul_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).mul_out(result, self, other);
}
static inline Tensor mul(const Tensor & self, const Tensor & other) {
    return infer_type(self).mul(self, other);
}
static inline Tensor & div_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).div_out(result, self, other);
}
static inline Tensor div(const Tensor & self, Scalar other) {
    return infer_type(self).div(self, other);
}
static inline Tensor & div_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).div_out(result, self, other);
}
static inline Tensor div(const Tensor & self, const Tensor & other) {
    return infer_type(self).div(self, other);
}
static inline Tensor & fmod_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).fmod_out(result, self, other);
}
static inline Tensor fmod(const Tensor & self, Scalar other) {
    return infer_type(self).fmod(self, other);
}
static inline Tensor & fmod_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).fmod_out(result, self, other);
}
static inline Tensor fmod(const Tensor & self, const Tensor & other) {
    return infer_type(self).fmod(self, other);
}
static inline Tensor & remainder_out(Tensor & result, const Tensor & self, Scalar other) {
    return infer_type(self).remainder_out(result, self, other);
}
static inline Tensor remainder(const Tensor & self, Scalar other) {
    return infer_type(self).remainder(self, other);
}
static inline Tensor & remainder_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).remainder_out(result, self, other);
}
static inline Tensor remainder(const Tensor & self, const Tensor & other) {
    return infer_type(self).remainder(self, other);
}
static inline Tensor & clamp_out(Tensor & destination, const Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp_out(destination, self, min, max);
}
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp(self, min, max);
}
static inline Tensor & clamp_out(Tensor & result, const Tensor & self, Scalar min) {
    return infer_type(self).clamp_out(result, self, min);
}
static inline Tensor clamp(const Tensor & self, Scalar min) {
    return infer_type(self).clamp(self, min);
}
static inline Scalar dot(const Tensor & self, const Tensor & tensor) {
    return infer_type(self).dot(self, tensor);
}
static inline Tensor & tril_out(Tensor & destination, const Tensor & self, int64_t diagonal) {
    return infer_type(self).tril_out(destination, self, diagonal);
}
static inline Tensor tril(const Tensor & self, int64_t diagonal) {
    return infer_type(self).tril(self, diagonal);
}
static inline Tensor & triu_out(Tensor & destination, const Tensor & self, int64_t diagonal) {
    return infer_type(self).triu_out(destination, self, diagonal);
}
static inline Tensor triu(const Tensor & self, int64_t diagonal) {
    return infer_type(self).triu(self, diagonal);
}
static inline Tensor & cross_out(Tensor & destination, const Tensor & self, const Tensor & other, int64_t dim) {
    return infer_type(self).cross_out(destination, self, other, dim);
}
static inline Tensor cross(const Tensor & self, const Tensor & other, int64_t dim) {
    return infer_type(self).cross(self, other, dim);
}
static inline Tensor & eye_out(Tensor & result, int64_t n, int64_t m) {
    return infer_type(result).eye_out(result, n, m);
}
static inline Tensor & diag_out(Tensor & result, const Tensor & self, int64_t diagonal) {
    return infer_type(self).diag_out(result, self, diagonal);
}
static inline Tensor diag(const Tensor & self, int64_t diagonal) {
    return infer_type(self).diag(self, diagonal);
}
static inline Tensor & addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).addmm_out(result, self, mat1, mat2, beta, alpha);
}
static inline Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).addmm(self, mat1, mat2, beta, alpha);
}
static inline Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self).addmv_out(result, self, mat, vec, beta, alpha);
}
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self).addmv(self, mat, vec, beta, alpha);
}
static inline Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self).addr_out(result, self, vec1, vec2, beta, alpha);
}
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self).addr(self, vec1, vec2, beta, alpha);
}
static inline Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) {
    return infer_type(self).ger_out(result, self, vec2);
}
static inline Tensor ger(const Tensor & self, const Tensor & vec2) {
    return infer_type(self).ger(self, vec2);
}
static inline Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec) {
    return infer_type(self).mv_out(result, self, vec);
}
static inline Tensor mv(const Tensor & self, const Tensor & vec) {
    return infer_type(self).mv(self, vec);
}
static inline Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) {
    return infer_type(self).mm_out(result, self, mat2);
}
static inline Tensor mm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self).mm(self, mat2);
}
static inline Tensor & bmm_out(Tensor & result, const Tensor & self, const Tensor & mat2) {
    return infer_type(self).bmm_out(result, self, mat2);
}
static inline Tensor bmm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self).bmm(self, mat2);
}
static inline Tensor & addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return infer_type(self).addbmm_out(result, self, batch1, batch2, beta, alpha);
}
static inline Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return infer_type(self).addbmm(self, batch1, batch2, beta, alpha);
}
static inline Tensor & baddbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return infer_type(self).baddbmm_out(result, self, batch1, batch2, beta, alpha);
}
static inline Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha) {
    return infer_type(self).baddbmm(self, batch1, batch2, beta, alpha);
}
static inline Tensor & addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return infer_type(self).addcmul_out(result, self, tensor1, tensor2, value);
}
static inline Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return infer_type(self).addcmul(self, tensor1, tensor2, value);
}
static inline Tensor & addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return infer_type(self).addcdiv_out(result, self, tensor1, tensor2, value);
}
static inline Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value) {
    return infer_type(self).addcdiv(self, tensor1, tensor2, value);
}
static inline std::tuple<Tensor &,Tensor &> gesv_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A) {
    return infer_type(self).gesv_out(solution, lu, self, A);
}
static inline std::tuple<Tensor,Tensor> gesv(const Tensor & self, const Tensor & A) {
    return infer_type(self).gesv(self, A);
}
static inline std::tuple<Tensor &,Tensor &> gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A) {
    return infer_type(self).gels_out(res1, res2, self, A);
}
static inline std::tuple<Tensor,Tensor> gels(const Tensor & self, const Tensor & A) {
    return infer_type(self).gels(self, A);
}
static inline std::tuple<Tensor &,Tensor &> trtrs_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
    return infer_type(self).trtrs_out(res1, res2, self, A, upper, transpose, unitriangular);
}
static inline std::tuple<Tensor,Tensor> trtrs(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular) {
    return infer_type(self).trtrs(self, A, upper, transpose, unitriangular);
}
static inline std::tuple<Tensor &,Tensor &> symeig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors, bool upper) {
    return infer_type(self).symeig_out(res1, res2, self, eigenvectors, upper);
}
static inline std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper) {
    return infer_type(self).symeig(self, eigenvectors, upper);
}
static inline std::tuple<Tensor &,Tensor &> eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors) {
    return infer_type(self).eig_out(res1, res2, self, eigenvectors);
}
static inline std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors) {
    return infer_type(self).eig(self, eigenvectors);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & res1, Tensor & res2, Tensor & res3, const Tensor & self, bool some) {
    return infer_type(self).svd_out(res1, res2, res3, self, some);
}
static inline std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some) {
    return infer_type(self).svd(self, some);
}
static inline Tensor & inverse_out(Tensor & output, const Tensor & self) {
    return infer_type(self).inverse_out(output, self);
}
static inline Tensor inverse(const Tensor & self) {
    return infer_type(self).inverse(self);
}
static inline Tensor & potrf_out(Tensor & output, const Tensor & self, bool upper) {
    return infer_type(self).potrf_out(output, self, upper);
}
static inline Tensor potrf(const Tensor & self, bool upper) {
    return infer_type(self).potrf(self, upper);
}
static inline Tensor & potrs_out(Tensor & result, const Tensor & self, const Tensor & input2, bool upper) {
    return infer_type(self).potrs_out(result, self, input2, upper);
}
static inline Tensor potrs(const Tensor & self, const Tensor & input2, bool upper) {
    return infer_type(self).potrs(self, input2, upper);
}
static inline Tensor & potri_out(Tensor & output, const Tensor & self, bool upper) {
    return infer_type(self).potri_out(output, self, upper);
}
static inline Tensor potri(const Tensor & self, bool upper) {
    return infer_type(self).potri(self, upper);
}
static inline std::tuple<Tensor &,Tensor &> pstrf_out(Tensor & res1, Tensor & res2, const Tensor & self, bool upper, Scalar tol) {
    return infer_type(self).pstrf_out(res1, res2, self, upper, tol);
}
static inline std::tuple<Tensor,Tensor> pstrf(const Tensor & self, bool upper, Scalar tol) {
    return infer_type(self).pstrf(self, upper, tol);
}
static inline std::tuple<Tensor &,Tensor &> qr_out(Tensor & res1, Tensor & res2, const Tensor & self) {
    return infer_type(self).qr_out(res1, res2, self);
}
static inline std::tuple<Tensor,Tensor> qr(const Tensor & self) {
    return infer_type(self).qr(self);
}
static inline std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self) {
    return infer_type(self).geqrf_out(res1, res2, self);
}
static inline std::tuple<Tensor,Tensor> geqrf(const Tensor & self) {
    return infer_type(self).geqrf(self);
}
static inline Tensor & orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2) {
    return infer_type(self).orgqr_out(result, self, input2);
}
static inline Tensor orgqr(const Tensor & self, const Tensor & input2) {
    return infer_type(self).orgqr(self, input2);
}
static inline Tensor & ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    return infer_type(self).ormqr_out(result, self, input2, input3, left, transpose);
}
static inline Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose) {
    return infer_type(self).ormqr(self, input2, input3, left, transpose);
}
static inline std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, const Tensor & info, bool pivot) {
    return infer_type(self).btrifact_out(result, pivots, self, info, pivot);
}
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self, const Tensor & info, bool pivot) {
    return infer_type(self).btrifact(self, info, pivot);
}
static inline Tensor & btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
    return infer_type(self).btrisolve_out(result, self, LU_data, LU_pivots);
}
static inline Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots) {
    return infer_type(self).btrisolve(self, LU_data, LU_pivots);
}
static inline Tensor & randperm_out(Tensor & result, int64_t n, Generator * generator) {
    return infer_type(result).randperm_out(result, n, generator);
}
static inline Tensor & multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
    return infer_type(self).multinomial_out(result, self, num_samples, replacement, generator);
}
static inline Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator) {
    return infer_type(self).multinomial(self, num_samples, replacement, generator);
}
static inline Tensor & normal_out(Tensor & output, const Tensor & means, double std, Generator * generator) {
    return infer_type(output).normal_out(output, means, std, generator);
}
static inline Tensor normal(const Tensor & means, double std, Generator * generator) {
    return infer_type(means).normal(means, std, generator);
}
static inline Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) {
    return infer_type(output).normal_out(output, mean, std, generator);
}
static inline Tensor normal(double mean, const Tensor & std, Generator * generator) {
    return infer_type(std).normal(mean, std, generator);
}
static inline Tensor & normal_out(Tensor & output, const Tensor & means, const Tensor & std, Generator * generator) {
    return infer_type(output).normal_out(output, means, std, generator);
}
static inline Tensor normal(const Tensor & means, const Tensor & std, Generator * generator) {
    return infer_type(means).normal(means, std, generator);
}
static inline Tensor & rand_out(Tensor & result, IntList size, Generator * generator) {
    return infer_type(result).rand_out(result, size, generator);
}
static inline Tensor & randn_out(Tensor & result, IntList size, Generator * generator) {
    return infer_type(result).randn_out(result, size, generator);
}
static inline Tensor & select_out(Tensor & result, const Tensor & self, int64_t dim, int64_t sliceIndex) {
    return infer_type(self).select_out(result, self, dim, sliceIndex);
}
static inline Tensor select(const Tensor & self, int64_t dim, int64_t sliceIndex) {
    return infer_type(self).select(self, dim, sliceIndex);
}
static inline Tensor & _unnarrow_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize) {
    return infer_type(self)._unnarrow_out(result, self, dimension, offset, dimSize);
}
static inline Tensor _unnarrow(const Tensor & self, int64_t dimension, int64_t offset, int64_t dimSize) {
    return infer_type(self)._unnarrow(self, dimension, offset, dimSize);
}
static inline Tensor & cat_out(Tensor & self, TensorList tensors, int64_t dim) {
    return infer_type(self).cat_out(self, tensors, dim);
}
static inline Tensor cat(TensorList tensors, int64_t dim) {
    return infer_type(tensors).cat(tensors, dim);
}
static inline Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(output).binary_cross_entropy_out(output, input, target, weight, size_average);
}
static inline Tensor binary_cross_entropy(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(input).binary_cross_entropy(input, target, weight, size_average);
}
static inline Tensor & binary_cross_entropy_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(output).binary_cross_entropy_forward_out(output, input, target, weight, size_average);
}
static inline Tensor binary_cross_entropy_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(input).binary_cross_entropy_forward(input, target, weight, size_average);
}
static inline Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(grad_input).binary_cross_entropy_backward_out(grad_input, input, target, weight, size_average);
}
static inline Tensor binary_cross_entropy_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average) {
    return infer_type(input).binary_cross_entropy_backward(input, target, weight, size_average);
}
static inline Tensor & kl_div_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).kl_div_out(output, input, target, size_average);
}
static inline Tensor kl_div(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).kl_div(input, target, size_average);
}
static inline Tensor & kl_div_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).kl_div_forward_out(output, input, target, size_average);
}
static inline Tensor kl_div_forward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).kl_div_forward(input, target, size_average);
}
static inline Tensor & kl_div_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(grad_input).kl_div_backward_out(grad_input, input, target, size_average);
}
static inline Tensor kl_div_backward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).kl_div_backward(input, target, size_average);
}
static inline Tensor & l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).l1_loss_out(output, input, target, size_average);
}
static inline Tensor l1_loss(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).l1_loss(input, target, size_average);
}
static inline Tensor & l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).l1_loss_forward_out(output, input, target, size_average);
}
static inline Tensor l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).l1_loss_forward(input, target, size_average);
}
static inline Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(grad_input).l1_loss_backward_out(grad_input, input, target, size_average);
}
static inline Tensor l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).l1_loss_backward(input, target, size_average);
}
static inline Tensor & mse_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(output).mse_loss_out(output, input, target, size_average, reduce);
}
static inline Tensor mse_loss(const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(input).mse_loss(input, target, size_average, reduce);
}
static inline Tensor & mse_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(output).mse_loss_forward_out(output, input, target, size_average, reduce);
}
static inline Tensor mse_loss_forward(const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(input).mse_loss_forward(input, target, size_average, reduce);
}
static inline Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(grad_input).mse_loss_backward_out(grad_input, grad_output, input, target, size_average, reduce);
}
static inline Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & input, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(grad_output).mse_loss_backward(grad_output, input, target, size_average, reduce);
}
static inline Tensor & multi_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(output).multi_margin_loss_out(output, input, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(input).multi_margin_loss(input, target, p, margin, weight, size_average);
}
static inline Tensor & multi_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(output).multi_margin_loss_forward_out(output, input, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss_forward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(input).multi_margin_loss_forward(input, target, p, margin, weight, size_average);
}
static inline Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(grad_input).multi_margin_loss_backward_out(grad_input, input, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss_backward(const Tensor & input, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(input).multi_margin_loss_backward(input, target, p, margin, weight, size_average);
}
static inline Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).multilabel_margin_loss_out(output, input, target, size_average);
}
static inline Tensor multilabel_margin_loss(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).multilabel_margin_loss(input, target, size_average);
}
static inline Tensor & multilabel_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) {
    return infer_type(output).multilabel_margin_loss_forward_out(output, input, target, size_average, is_target);
}
static inline Tensor multilabel_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) {
    return infer_type(input).multilabel_margin_loss_forward(input, target, size_average, is_target);
}
static inline Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) {
    return infer_type(grad_input).multilabel_margin_loss_backward_out(grad_input, input, target, size_average, is_target);
}
static inline Tensor multilabel_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average, const Tensor & is_target) {
    return infer_type(input).multilabel_margin_loss_backward(input, target, size_average, is_target);
}
static inline Tensor & nll_loss_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index) {
    return infer_type(output).nll_loss_out(output, input, target, weight, size_average, ignore_index);
}
static inline Tensor nll_loss(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index) {
    return infer_type(input).nll_loss(input, target, weight, size_average, ignore_index);
}
static inline Tensor & nll_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(output).nll_loss_forward_out(output, input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor nll_loss_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(input).nll_loss_forward(input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(grad_input).nll_loss_backward_out(grad_input, input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor nll_loss_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(input).nll_loss_backward(input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor & nll_loss2d_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index) {
    return infer_type(output).nll_loss2d_out(output, input, target, weight, size_average, ignore_index);
}
static inline Tensor nll_loss2d(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index) {
    return infer_type(input).nll_loss2d(input, target, weight, size_average, ignore_index);
}
static inline Tensor & nll_loss2d_forward_out(Tensor & output, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(output).nll_loss2d_forward_out(output, input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor nll_loss2d_forward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(input).nll_loss2d_forward(input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(grad_input).nll_loss2d_backward_out(grad_input, input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor nll_loss2d_backward(const Tensor & input, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, const Tensor & total_weight) {
    return infer_type(input).nll_loss2d_backward(input, target, weight, size_average, ignore_index, total_weight);
}
static inline Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).smooth_l1_loss_out(output, input, target, size_average);
}
static inline Tensor smooth_l1_loss(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).smooth_l1_loss(input, target, size_average);
}
static inline Tensor & smooth_l1_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).smooth_l1_loss_forward_out(output, input, target, size_average);
}
static inline Tensor smooth_l1_loss_forward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).smooth_l1_loss_forward(input, target, size_average);
}
static inline Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(grad_input).smooth_l1_loss_backward_out(grad_input, input, target, size_average);
}
static inline Tensor smooth_l1_loss_backward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).smooth_l1_loss_backward(input, target, size_average);
}
static inline Tensor & soft_margin_loss_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).soft_margin_loss_out(output, input, target, size_average);
}
static inline Tensor soft_margin_loss(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).soft_margin_loss(input, target, size_average);
}
static inline Tensor & soft_margin_loss_forward_out(Tensor & output, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(output).soft_margin_loss_forward_out(output, input, target, size_average);
}
static inline Tensor soft_margin_loss_forward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).soft_margin_loss_forward(input, target, size_average);
}
static inline Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(grad_input).soft_margin_loss_backward_out(grad_input, input, target, size_average);
}
static inline Tensor soft_margin_loss_backward(const Tensor & input, const Tensor & target, bool size_average) {
    return infer_type(input).soft_margin_loss_backward(input, target, size_average);
}
static inline Tensor & elu_out(Tensor & output, const Tensor & input, Scalar alpha, bool inplace) {
    return infer_type(output).elu_out(output, input, alpha, inplace);
}
static inline Tensor elu(const Tensor & input, Scalar alpha, bool inplace) {
    return infer_type(input).elu(input, alpha, inplace);
}
static inline Tensor & elu_forward_out(Tensor & output, const Tensor & input, Scalar alpha, bool inplace) {
    return infer_type(output).elu_forward_out(output, input, alpha, inplace);
}
static inline Tensor elu_forward(const Tensor & input, Scalar alpha, bool inplace) {
    return infer_type(input).elu_forward(input, alpha, inplace);
}
static inline Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output) {
    return infer_type(grad_input).elu_backward_out(grad_input, grad_output, input, alpha, inplace, output);
}
static inline Tensor elu_backward(const Tensor & grad_output, const Tensor & input, Scalar alpha, bool inplace, const Tensor & output) {
    return infer_type(grad_output).elu_backward(grad_output, input, alpha, inplace, output);
}
static inline Tensor & glu_out(Tensor & output, const Tensor & input, int64_t dim) {
    return infer_type(output).glu_out(output, input, dim);
}
static inline Tensor glu(const Tensor & input, int64_t dim) {
    return infer_type(input).glu(input, dim);
}
static inline Tensor & glu_forward_out(Tensor & output, const Tensor & input, int64_t dim) {
    return infer_type(output).glu_forward_out(output, input, dim);
}
static inline Tensor glu_forward(const Tensor & input, int64_t dim) {
    return infer_type(input).glu_forward(input, dim);
}
static inline Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, int64_t dim) {
    return infer_type(grad_input).glu_backward_out(grad_input, grad_output, input, dim);
}
static inline Tensor glu_backward(const Tensor & grad_output, const Tensor & input, int64_t dim) {
    return infer_type(grad_output).glu_backward(grad_output, input, dim);
}
static inline Tensor & hardshrink_out(Tensor & output, const Tensor & input, Scalar lambd) {
    return infer_type(output).hardshrink_out(output, input, lambd);
}
static inline Tensor hardshrink(const Tensor & input, Scalar lambd) {
    return infer_type(input).hardshrink(input, lambd);
}
static inline Tensor & hardshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd) {
    return infer_type(output).hardshrink_forward_out(output, input, lambd);
}
static inline Tensor hardshrink_forward(const Tensor & input, Scalar lambd) {
    return infer_type(input).hardshrink_forward(input, lambd);
}
static inline Tensor & hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd) {
    return infer_type(grad_input).hardshrink_backward_out(grad_input, grad_output, input, lambd);
}
static inline Tensor hardshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd) {
    return infer_type(grad_output).hardshrink_backward(grad_output, input, lambd);
}
static inline Tensor & hardtanh_out(Tensor & output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(output).hardtanh_out(output, input, min_val, max_val, inplace);
}
static inline Tensor hardtanh(const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(input).hardtanh(input, min_val, max_val, inplace);
}
static inline Tensor & hardtanh_forward_out(Tensor & output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(output).hardtanh_forward_out(output, input, min_val, max_val, inplace);
}
static inline Tensor hardtanh_forward(const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(input).hardtanh_forward(input, min_val, max_val, inplace);
}
static inline Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(grad_input).hardtanh_backward_out(grad_input, grad_output, input, min_val, max_val, inplace);
}
static inline Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & input, Scalar min_val, Scalar max_val, bool inplace) {
    return infer_type(grad_output).hardtanh_backward(grad_output, input, min_val, max_val, inplace);
}
static inline Tensor & leaky_relu_out(Tensor & output, const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(output).leaky_relu_out(output, input, negative_slope, inplace);
}
static inline Tensor leaky_relu(const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(input).leaky_relu(input, negative_slope, inplace);
}
static inline Tensor & leaky_relu_forward_out(Tensor & output, const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(output).leaky_relu_forward_out(output, input, negative_slope, inplace);
}
static inline Tensor leaky_relu_forward(const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(input).leaky_relu_forward(input, negative_slope, inplace);
}
static inline Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(grad_input).leaky_relu_backward_out(grad_input, grad_output, input, negative_slope, inplace);
}
static inline Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & input, Scalar negative_slope, bool inplace) {
    return infer_type(grad_output).leaky_relu_backward(grad_output, input, negative_slope, inplace);
}
static inline Tensor & log_sigmoid_out(Tensor & output, const Tensor & input) {
    return infer_type(output).log_sigmoid_out(output, input);
}
static inline Tensor log_sigmoid(const Tensor & input) {
    return infer_type(input).log_sigmoid(input);
}
static inline Tensor & log_sigmoid_forward_out(Tensor & output, const Tensor & input, const Tensor & buffer) {
    return infer_type(output).log_sigmoid_forward_out(output, input, buffer);
}
static inline Tensor log_sigmoid_forward(const Tensor & input, const Tensor & buffer) {
    return infer_type(input).log_sigmoid_forward(input, buffer);
}
static inline Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & buffer) {
    return infer_type(grad_input).log_sigmoid_backward_out(grad_input, grad_output, input, buffer);
}
static inline Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & input, const Tensor & buffer) {
    return infer_type(grad_output).log_sigmoid_backward(grad_output, input, buffer);
}
static inline Tensor & log_softmax_out(Tensor & output, const Tensor & input, int64_t dim) {
    return infer_type(output).log_softmax_out(output, input, dim);
}
static inline Tensor log_softmax(const Tensor & input, int64_t dim) {
    return infer_type(input).log_softmax(input, dim);
}
static inline Tensor & log_softmax_forward_out(Tensor & output, const Tensor & input) {
    return infer_type(output).log_softmax_forward_out(output, input);
}
static inline Tensor log_softmax_forward(const Tensor & input) {
    return infer_type(input).log_softmax_forward(input);
}
static inline Tensor & log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output) {
    return infer_type(grad_input).log_softmax_backward_out(grad_input, grad_output, input, output);
}
static inline Tensor log_softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output) {
    return infer_type(grad_output).log_softmax_backward(grad_output, input, output);
}
static inline Tensor & prelu_out(Tensor & output, const Tensor & input, const Tensor & weight) {
    return infer_type(output).prelu_out(output, input, weight);
}
static inline Tensor prelu(const Tensor & input, const Tensor & weight) {
    return infer_type(input).prelu(input, weight);
}
static inline Tensor & prelu_forward_out(Tensor & output, const Tensor & input, const Tensor & weight) {
    return infer_type(output).prelu_forward_out(output, input, weight);
}
static inline Tensor prelu_forward(const Tensor & input, const Tensor & weight) {
    return infer_type(input).prelu_forward(input, weight);
}
static inline std::tuple<Tensor &,Tensor &> prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & input, const Tensor & weight) {
    return infer_type(grad_input).prelu_backward_out(grad_input, grad_weight, grad_output, input, weight);
}
static inline std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, std::array<bool, 2> output_mask) {
    return infer_type(grad_output).prelu_backward(grad_output, input, weight, output_mask);
}
static inline Tensor & rrelu_out(Tensor & output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator) {
    return infer_type(output).rrelu_out(output, input, lower, upper, training, inplace, generator);
}
static inline Tensor rrelu(const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator) {
    return infer_type(input).rrelu(input, lower, upper, training, inplace, generator);
}
static inline Tensor & rrelu_forward_out(Tensor & output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise) {
    return infer_type(output).rrelu_forward_out(output, input, lower, upper, training, inplace, generator, noise);
}
static inline Tensor rrelu_forward(const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, Generator * generator, const Tensor & noise) {
    return infer_type(input).rrelu_forward(input, lower, upper, training, inplace, generator, noise);
}
static inline Tensor & rrelu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise) {
    return infer_type(grad_input).rrelu_backward_out(grad_input, grad_output, input, lower, upper, training, inplace, noise);
}
static inline Tensor rrelu_backward(const Tensor & grad_output, const Tensor & input, Scalar lower, Scalar upper, bool training, bool inplace, const Tensor & noise) {
    return infer_type(grad_output).rrelu_backward(grad_output, input, lower, upper, training, inplace, noise);
}
static inline Tensor & softmax_out(Tensor & output, const Tensor & input, int64_t dim) {
    return infer_type(output).softmax_out(output, input, dim);
}
static inline Tensor softmax(const Tensor & input, int64_t dim) {
    return infer_type(input).softmax(input, dim);
}
static inline Tensor & softmax_forward_out(Tensor & output, const Tensor & input) {
    return infer_type(output).softmax_forward_out(output, input);
}
static inline Tensor softmax_forward(const Tensor & input) {
    return infer_type(input).softmax_forward(input);
}
static inline Tensor & softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & output) {
    return infer_type(grad_input).softmax_backward_out(grad_input, grad_output, input, output);
}
static inline Tensor softmax_backward(const Tensor & grad_output, const Tensor & input, const Tensor & output) {
    return infer_type(grad_output).softmax_backward(grad_output, input, output);
}
static inline Tensor & softplus_out(Tensor & output, const Tensor & input, Scalar beta, Scalar threshold) {
    return infer_type(output).softplus_out(output, input, beta, threshold);
}
static inline Tensor softplus(const Tensor & input, Scalar beta, Scalar threshold) {
    return infer_type(input).softplus(input, beta, threshold);
}
static inline Tensor & softplus_forward_out(Tensor & output, const Tensor & input, Scalar beta, Scalar threshold) {
    return infer_type(output).softplus_forward_out(output, input, beta, threshold);
}
static inline Tensor softplus_forward(const Tensor & input, Scalar beta, Scalar threshold) {
    return infer_type(input).softplus_forward(input, beta, threshold);
}
static inline Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output) {
    return infer_type(grad_input).softplus_backward_out(grad_input, grad_output, input, beta, threshold, output);
}
static inline Tensor softplus_backward(const Tensor & grad_output, const Tensor & input, Scalar beta, Scalar threshold, const Tensor & output) {
    return infer_type(grad_output).softplus_backward(grad_output, input, beta, threshold, output);
}
static inline Tensor & softshrink_out(Tensor & output, const Tensor & input, Scalar lambd) {
    return infer_type(output).softshrink_out(output, input, lambd);
}
static inline Tensor softshrink(const Tensor & input, Scalar lambd) {
    return infer_type(input).softshrink(input, lambd);
}
static inline Tensor & softshrink_forward_out(Tensor & output, const Tensor & input, Scalar lambd) {
    return infer_type(output).softshrink_forward_out(output, input, lambd);
}
static inline Tensor softshrink_forward(const Tensor & input, Scalar lambd) {
    return infer_type(input).softshrink_forward(input, lambd);
}
static inline Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar lambd) {
    return infer_type(grad_input).softshrink_backward_out(grad_input, grad_output, input, lambd);
}
static inline Tensor softshrink_backward(const Tensor & grad_output, const Tensor & input, Scalar lambd) {
    return infer_type(grad_output).softshrink_backward(grad_output, input, lambd);
}
static inline Tensor & threshold_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(output).threshold_out(output, input, threshold, value, inplace);
}
static inline Tensor threshold(const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(input).threshold(input, threshold, value, inplace);
}
static inline Tensor & threshold_forward_out(Tensor & output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(output).threshold_forward_out(output, input, threshold, value, inplace);
}
static inline Tensor threshold_forward(const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(input).threshold_forward(input, threshold, value, inplace);
}
static inline Tensor & threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(grad_input).threshold_backward_out(grad_input, grad_output, input, threshold, value, inplace);
}
static inline Tensor threshold_backward(const Tensor & grad_output, const Tensor & input, Scalar threshold, Scalar value, bool inplace) {
    return infer_type(grad_output).threshold_backward(grad_output, input, threshold, value, inplace);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size) {
    return infer_type(output).adaptive_max_pool2d_out(output, indices, input, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & input, IntList output_size) {
    return infer_type(input).adaptive_max_pool2d(input, output_size);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList output_size) {
    return infer_type(output).adaptive_max_pool2d_forward_out(output, indices, input, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d_forward(const Tensor & input, IntList output_size) {
    return infer_type(input).adaptive_max_pool2d_forward(input, output_size);
}
static inline Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices) {
    return infer_type(grad_input).adaptive_max_pool2d_backward_out(grad_input, grad_output, input, indices);
}
static inline Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices) {
    return infer_type(grad_output).adaptive_max_pool2d_backward(grad_output, input, indices);
}
static inline Tensor & avg_pool2d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(output).avg_pool2d_out(output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool2d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(output).avg_pool2d_forward_out(output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).avg_pool2d_forward(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(grad_input).avg_pool2d_backward_out(grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(grad_output).avg_pool2d_backward(grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(output).avg_pool3d_out(output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_forward_out(Tensor & output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(output).avg_pool3d_forward_out(output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(input).avg_pool3d_forward(input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(grad_input).avg_pool3d_backward_out(grad_input, grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(grad_output).avg_pool3d_backward(grad_output, input, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline std::tuple<Tensor &,Tensor &> max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(output).max_pool2d_out(output, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool2d(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(input).max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor &,Tensor &> max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(output).max_pool2d_forward_out(output, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool2d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(input).max_pool2d_forward(input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline Tensor & max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(grad_input).max_pool2d_backward_out(grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor max_pool2d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(grad_output).max_pool2d_backward(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline std::tuple<Tensor &,Tensor &> max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(output).max_pool3d_out(output, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool3d(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(input).max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor &,Tensor &> max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(output).max_pool3d_forward_out(output, indices, input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool3d_forward(const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(input).max_pool3d_forward(input, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline Tensor & max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(grad_input).max_pool3d_backward_out(grad_input, grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor max_pool3d_backward(const Tensor & grad_output, const Tensor & input, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(grad_output).max_pool3d_backward(grad_output, input, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor & max_unpool2d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(output).max_unpool2d_out(output, input, indices, output_size);
}
static inline Tensor max_unpool2d(const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(input).max_unpool2d(input, indices, output_size);
}
static inline Tensor & max_unpool2d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(output).max_unpool2d_forward_out(output, input, indices, output_size);
}
static inline Tensor max_unpool2d_forward(const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(input).max_unpool2d_forward(input, indices, output_size);
}
static inline Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(grad_input).max_unpool2d_backward_out(grad_input, grad_output, input, indices, output_size);
}
static inline Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size) {
    return infer_type(grad_output).max_unpool2d_backward(grad_output, input, indices, output_size);
}
static inline Tensor & max_unpool3d_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(output).max_unpool3d_out(output, input, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(input).max_unpool3d(input, indices, output_size, stride, padding);
}
static inline Tensor & max_unpool3d_forward_out(Tensor & output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(output).max_unpool3d_forward_out(output, input, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d_forward(const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(input).max_unpool3d_forward(input, indices, output_size, stride, padding);
}
static inline Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(grad_input).max_unpool3d_backward_out(grad_input, grad_output, input, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(grad_output).max_unpool3d_backward(grad_output, input, indices, output_size, stride, padding);
}
static inline Tensor & _sigmoid_out(Tensor & output, const Tensor & input) {
    return infer_type(output)._sigmoid_out(output, input);
}
static inline Tensor _sigmoid(const Tensor & input) {
    return infer_type(input)._sigmoid(input);
}
static inline Tensor & _sigmoid_forward_out(Tensor & output, const Tensor & input) {
    return infer_type(output)._sigmoid_forward_out(output, input);
}
static inline Tensor _sigmoid_forward(const Tensor & input) {
    return infer_type(input)._sigmoid_forward(input);
}
static inline Tensor & _sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_input)._sigmoid_backward_out(grad_input, grad_output, output);
}
static inline Tensor _sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_output)._sigmoid_backward(grad_output, output);
}
static inline Tensor & _tanh_out(Tensor & output, const Tensor & input) {
    return infer_type(output)._tanh_out(output, input);
}
static inline Tensor _tanh(const Tensor & input) {
    return infer_type(input)._tanh(input);
}
static inline Tensor & _tanh_forward_out(Tensor & output, const Tensor & input) {
    return infer_type(output)._tanh_forward_out(output, input);
}
static inline Tensor _tanh_forward(const Tensor & input) {
    return infer_type(input)._tanh_forward(input);
}
static inline Tensor & _tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_input)._tanh_backward_out(grad_input, grad_output, output);
}
static inline Tensor _tanh_backward(const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_output)._tanh_backward(grad_output, output);
}
static inline Tensor & batch_norm_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(output).batch_norm_out(output, input, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(input).batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline Tensor & batch_norm_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) {
    return infer_type(output).batch_norm_forward_out(output, input, weight, bias, running_mean, running_var, training, momentum, eps, save_mean, save_std);
}
static inline Tensor batch_norm_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, const Tensor & save_mean, const Tensor & save_std) {
    return infer_type(input).batch_norm_forward(input, weight, bias, running_mean, running_var, training, momentum, eps, save_mean, save_std);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) {
    return infer_type(grad_input).batch_norm_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, running_mean, running_var, training, eps, save_mean, save_std);
}
static inline std::tuple<Tensor,Tensor,Tensor> batch_norm_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).batch_norm_backward(grad_output, input, weight, running_mean, running_var, training, eps, save_mean, save_std, output_mask);
}
static inline Tensor & conv_transpose2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(output).conv_transpose2d_out(output, input, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(input).conv_transpose2d(input, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline Tensor & conv_transpose2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(output).conv_transpose2d_forward_out(output, input, weight, kernel_size, bias, stride, padding, output_padding, dilation, columns, ones);
}
static inline Tensor conv_transpose2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(input).conv_transpose2d_forward(input, weight, kernel_size, bias, stride, padding, output_padding, dilation, columns, ones);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(grad_input).conv_transpose2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv_transpose2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv_transpose2d_backward(grad_output, input, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
static inline Tensor & conv_transpose3d_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(output).conv_transpose3d_out(output, input, weight, bias, stride, padding, output_padding, dilation);
}
static inline Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(input).conv_transpose3d(input, weight, bias, stride, padding, output_padding, dilation);
}
static inline Tensor & conv_transpose3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(output).conv_transpose3d_forward_out(output, input, weight, bias, stride, padding, output_padding, dilation, finput, fgrad_input);
}
static inline Tensor conv_transpose3d_forward(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(input).conv_transpose3d_forward(input, weight, bias, stride, padding, output_padding, dilation, finput, fgrad_input);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(grad_input).conv_transpose3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, stride, padding, output_padding, dilation, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv_transpose3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv_transpose3d_backward(grad_output, input, weight, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
static inline Tensor & conv2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(output).conv2d_out(output, input, weight, kernel_size, bias, stride, padding);
}
static inline Tensor conv2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(input).conv2d(input, weight, kernel_size, bias, stride, padding);
}
static inline Tensor & conv2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(output).conv2d_forward_out(output, input, weight, kernel_size, bias, stride, padding, finput, fgrad_input);
}
static inline Tensor conv2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(input).conv2d_forward(input, weight, kernel_size, bias, stride, padding, finput, fgrad_input);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(grad_input).conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride, padding, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv2d_backward(grad_output, input, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
static inline Tensor & conv3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(output).conv3d_out(output, input, weight, kernel_size, bias, stride, padding);
}
static inline Tensor conv3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(input).conv3d(input, weight, kernel_size, bias, stride, padding);
}
static inline Tensor & conv3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) {
    return infer_type(output).conv3d_forward_out(output, input, weight, kernel_size, bias, stride, padding, finput);
}
static inline Tensor conv3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, const Tensor & finput) {
    return infer_type(input).conv3d_forward(input, weight, kernel_size, bias, stride, padding, finput);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(grad_input).conv3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride, padding, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv3d_backward(grad_output, input, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
static inline Tensor & conv_dilated2d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(output).conv_dilated2d_out(output, input, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor conv_dilated2d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(input).conv_dilated2d(input, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor & conv_dilated2d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(output).conv_dilated2d_forward_out(output, input, weight, kernel_size, bias, stride, padding, dilation, columns, ones);
}
static inline Tensor conv_dilated2d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(input).conv_dilated2d_forward(input, weight, kernel_size, bias, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(grad_input).conv_dilated2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv_dilated2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv_dilated2d_backward(grad_output, input, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
static inline Tensor & conv_dilated3d_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(output).conv_dilated3d_out(output, input, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor conv_dilated3d(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(input).conv_dilated3d(input, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor & conv_dilated3d_forward_out(Tensor & output, const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(output).conv_dilated3d_forward_out(output, input, weight, kernel_size, bias, stride, padding, dilation, columns, ones);
}
static inline Tensor conv_dilated3d_forward(const Tensor & input, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(input).conv_dilated3d_forward(input, weight, kernel_size, bias, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(grad_input).conv_dilated3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, input, weight, kernel_size, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv_dilated3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool, 3> output_mask) {
    return infer_type(grad_output).conv_dilated3d_backward(grad_output, input, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
static inline std::vector<Tensor> split(Tensor self, int64_t split_size, int64_t dim) {
    return infer_type(self).split(self, split_size, dim);
}
static inline std::vector<Tensor> chunk(Tensor self, int64_t chunks, int64_t dim) {
    return infer_type(self).chunk(self, chunks, dim);
}

}
