#pragma once

#include "ATen/Scalar.h"
#include "ATen/Type.h"
#include "ATen/Tensor.h"
#include "ATen/Storage.h"
#include "ATen/Generator.h"


namespace at {

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
static inline Tensor & nonzero_out(Tensor & result, const Tensor & self);
static inline Tensor nonzero(const Tensor & self);
static inline Tensor & index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index);
static inline Tensor & take_out(Tensor & result, const Tensor & self, const Tensor & index);
static inline Tensor take(const Tensor & self, const Tensor & index);
static inline Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
static inline Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step=1);
static inline Tensor & arange_out(Tensor & result, Scalar end);
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
static inline Tensor min(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor & max_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor max(const Tensor & self, const Tensor & other);
static inline Tensor max(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim=-1, bool keepdim=false);
static inline std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim=false);
static inline std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor median(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim=-1, bool descending=false);
static inline std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim=-1, bool descending=false);
static inline std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true);
static inline std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim=-1, bool largest=true, bool sorted=true);
static inline Tensor & abs_out(Tensor & result, const Tensor & self);
static inline Tensor abs(const Tensor & self);
static inline Tensor & sigmoid_out(Tensor & result, const Tensor & self);
static inline Tensor sigmoid(const Tensor & self);
static inline Tensor & log_out(Tensor & result, const Tensor & self);
static inline Tensor log(const Tensor & self);
static inline Tensor & log1p_out(Tensor & result, const Tensor & self);
static inline Tensor log1p(const Tensor & self);
static inline Tensor & lgamma_out(Tensor & result, const Tensor & self);
static inline Tensor lgamma(const Tensor & self);
static inline Tensor & digamma_out(Tensor & result, const Tensor & self);
static inline Tensor digamma(const Tensor & self);
static inline Tensor & polygamma_out(Tensor & result, int64_t n, const Tensor & self);
static inline Tensor polygamma(int64_t n, const Tensor & self);
static inline Tensor & exp_out(Tensor & result, const Tensor & self);
static inline Tensor exp(const Tensor & self);
static inline Tensor & expm1_out(Tensor & result, const Tensor & self);
static inline Tensor expm1(const Tensor & self);
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
static inline Tensor & mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor mean(const Tensor & self);
static inline Tensor & var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor var(const Tensor & self, bool unbiased=true);
static inline Tensor & std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased=true, bool keepdim=false);
static inline Tensor std(const Tensor & self, bool unbiased=true);
static inline Tensor & norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim=false);
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim=false);
static inline Tensor norm(const Tensor & self, Scalar p=2);
static inline Tensor & renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
static inline Tensor dist(const Tensor & self, const Tensor & other, Scalar p=2);
static inline Tensor & reciprocal_out(Tensor & result, const Tensor & self);
static inline Tensor reciprocal(const Tensor & self);
static inline Tensor & neg_out(Tensor & result, const Tensor & self);
static inline Tensor neg(const Tensor & self);
static inline Tensor & atan2_out(Tensor & result, const Tensor & self, const Tensor & other);
static inline Tensor atan2(const Tensor & self, const Tensor & other);
static inline Tensor & pow_out(Tensor & result, const Tensor & self, Scalar exponent);
static inline Tensor pow(const Tensor & self, Scalar exponent);
static inline Tensor & pow_out(Tensor & result, const Tensor & self, const Tensor & exponent);
static inline Tensor pow(const Tensor & self, const Tensor & exponent);
static inline Tensor & pow_out(Tensor & result, Scalar base, const Tensor & self);
static inline Tensor pow(Scalar base, const Tensor & self);
static inline Tensor & lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight);
static inline Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight);
static inline Tensor & linspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
static inline Tensor & logspace_out(Tensor & result, Scalar start, Scalar end, int64_t steps=100);
static inline Tensor & histc_out(Tensor & result, const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0);
static inline Tensor histc(const Tensor & self, int64_t bins=100, Scalar min=0, Scalar max=0);
static inline Tensor & sum_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor sum(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor sum(const Tensor & self);
static inline Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim=false);
static inline Tensor prod(const Tensor & self);
static inline Tensor & cumsum_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor cumsum(const Tensor & self, int64_t dim);
static inline Tensor & cumprod_out(Tensor & result, const Tensor & self, int64_t dim);
static inline Tensor cumprod(const Tensor & self, int64_t dim);
static inline Tensor & sign_out(Tensor & result, const Tensor & self);
static inline Tensor sign(const Tensor & self);
static inline Tensor trace(const Tensor & self);
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
static inline Tensor & clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max);
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max);
static inline Tensor & clamp_(Tensor & self, Scalar min, Scalar max);
static inline Tensor & clamp_min_out(Tensor & result, const Tensor & self, Scalar min);
static inline Tensor clamp_min(const Tensor & self, Scalar min);
static inline Tensor & clamp_min_(Tensor & self, Scalar min);
static inline Tensor & clamp_max_out(Tensor & result, const Tensor & self, Scalar max);
static inline Tensor clamp_max(const Tensor & self, Scalar max);
static inline Tensor & clamp_max_(Tensor & self, Scalar max);
static inline Tensor _dot(const Tensor & self, const Tensor & tensor);
static inline Tensor & tril_out(Tensor & result, const Tensor & self, int64_t diagonal=0);
static inline Tensor tril(const Tensor & self, int64_t diagonal=0);
static inline Tensor & triu_out(Tensor & result, const Tensor & self, int64_t diagonal=0);
static inline Tensor triu(const Tensor & self, int64_t diagonal=0);
static inline Tensor & cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim=-1);
static inline Tensor cross(const Tensor & self, const Tensor & other, int64_t dim=-1);
static inline Tensor & eye_out(Tensor & result, int64_t n, int64_t m=-1);
static inline Tensor & diag_out(Tensor & result, const Tensor & self, int64_t diagonal=0);
static inline Tensor diag(const Tensor & self, int64_t diagonal=0);
static inline Tensor & addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & _addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor _addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor & _addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & _ger_out(Tensor & result, const Tensor & self, const Tensor & vec2);
static inline Tensor _ger(const Tensor & self, const Tensor & vec2);
static inline Tensor & _mv_out(Tensor & result, const Tensor & self, const Tensor & vec);
static inline Tensor _mv(const Tensor & self, const Tensor & vec);
static inline Tensor & _mm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
static inline Tensor _mm(const Tensor & self, const Tensor & mat2);
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
static inline std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot=true);
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self, bool pivot=true);
static inline std::tuple<Tensor &,Tensor &,Tensor &> btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot=true);
static inline std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(const Tensor & self, bool pivot=true);
static inline Tensor & btrisolve_out(Tensor & result, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots);
static inline Tensor btrisolve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots);
static inline Tensor & randperm_out(Tensor & result, int64_t n, Generator * generator=nullptr);
static inline Tensor & multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr);
static inline Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement=false, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, const Tensor & mean, double std=1, Generator * generator=nullptr);
static inline Tensor normal(const Tensor & mean, double std=1, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor normal(double mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor & normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor normal(const Tensor & mean, const Tensor & std, Generator * generator=nullptr);
static inline Tensor & rand_out(Tensor & result, IntList size, Generator * generator=nullptr);
static inline Tensor & randn_out(Tensor & result, IntList size, Generator * generator=nullptr);
static inline Tensor & bernoulli_out(Tensor & output, const Tensor & self, Generator * generator=nullptr);
static inline Tensor bernoulli(const Tensor & self, Generator * generator=nullptr);
static inline Tensor & _standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator=nullptr);
static inline Tensor _standard_gamma(const Tensor & self, Generator * generator=nullptr);
static inline Tensor & _dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total);
static inline Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total);
static inline Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size);
static inline Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values);
static inline Tensor alias(const Tensor & self);
static inline Tensor & as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset=-1);
static inline Tensor as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset=-1);
static inline Tensor & as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset=-1);
static inline Tensor & _cat_out(Tensor & self, TensorList tensors, int64_t dim=0);
static inline Tensor _cat(TensorList tensors, int64_t dim=0);
static inline Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, bool reduce=true);
static inline Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, bool reduce=true);
static inline Tensor & binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce);
static inline Tensor binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce);
static inline Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce);
static inline Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce);
static inline Tensor & kl_div_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor kl_div(const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor & kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor l1_loss(const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor & l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor mse_loss(const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor & mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true);
static inline Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p=1, Scalar margin=1, const Tensor & weight={}, bool size_average=true);
static inline Tensor & multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor multi_margin_loss_backward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average);
static inline Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target);
static inline Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target);
static inline Tensor & nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100, bool reduce=true);
static inline Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100, bool reduce=true);
static inline std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce);
static inline std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce);
static inline Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight);
static inline Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight);
static inline Tensor & nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100, bool reduce=true);
static inline Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight={}, bool size_average=true, int64_t ignore_index=-100, bool reduce=true);
static inline std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce);
static inline std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce);
static inline Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight);
static inline Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight);
static inline Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average=true, bool reduce=true);
static inline Tensor & smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce);
static inline Tensor & soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average=true);
static inline Tensor soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average=true);
static inline Tensor & soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average);
static inline Tensor soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average);
static inline Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, bool size_average);
static inline Tensor soft_margin_loss_backward(const Tensor & self, const Tensor & target, bool size_average);
static inline Tensor & elu_out(Tensor & output, const Tensor & self, Scalar alpha=1, Scalar scale=1);
static inline Tensor elu(const Tensor & self, Scalar alpha=1, Scalar scale=1);
static inline Tensor & elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale);
static inline Tensor elu_forward(const Tensor & self, Scalar alpha, Scalar scale);
static inline Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output);
static inline Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output);
static inline Tensor & elu_(Tensor & self, Scalar alpha=1, Scalar scale=1);
static inline Tensor & elu_forward_(Tensor & self, Scalar alpha, Scalar scale);
static inline Tensor & glu_out(Tensor & output, const Tensor & self, int64_t dim=-1);
static inline Tensor glu(const Tensor & self, int64_t dim=-1);
static inline Tensor & glu_forward_out(Tensor & output, const Tensor & self, int64_t dim);
static inline Tensor glu_forward(const Tensor & self, int64_t dim);
static inline Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim);
static inline Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim);
static inline Tensor & hardshrink_out(Tensor & output, const Tensor & self, Scalar lambd=0.5);
static inline Tensor hardshrink(const Tensor & self, Scalar lambd=0.5);
static inline Tensor & hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd);
static inline Tensor hardshrink_forward(const Tensor & self, Scalar lambd);
static inline Tensor & hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd);
static inline Tensor hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd);
static inline Tensor & hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val=-1, Scalar max_val=1);
static inline Tensor hardtanh(const Tensor & self, Scalar min_val=-1, Scalar max_val=1);
static inline Tensor & hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val);
static inline Tensor hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val);
static inline Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val);
static inline Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val);
static inline Tensor & hardtanh_(Tensor & self, Scalar min_val=-1, Scalar max_val=1);
static inline Tensor & hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val);
static inline Tensor & leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope=0.01);
static inline Tensor leaky_relu(const Tensor & self, Scalar negative_slope=0.01);
static inline Tensor & leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope);
static inline Tensor leaky_relu_forward(const Tensor & self, Scalar negative_slope);
static inline Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope);
static inline Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope);
static inline Tensor & leaky_relu_(Tensor & self, Scalar negative_slope=0.01);
static inline Tensor & leaky_relu_forward_(Tensor & self, Scalar negative_slope);
static inline Tensor & log_sigmoid_out(Tensor & output, const Tensor & self);
static inline Tensor log_sigmoid(const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self);
static inline std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self);
static inline Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer);
static inline Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer);
static inline Tensor & log_softmax_out(Tensor & output, const Tensor & self, int64_t dim);
static inline Tensor log_softmax(const Tensor & self, int64_t dim);
static inline Tensor & log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim);
static inline Tensor log_softmax_forward(const Tensor & self, int64_t dim);
static inline Tensor & log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output);
static inline Tensor log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output);
static inline Tensor & prelu_out(Tensor & output, const Tensor & self, const Tensor & weight);
static inline Tensor prelu(const Tensor & self, const Tensor & weight);
static inline Tensor & prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight);
static inline Tensor prelu_forward(const Tensor & self, const Tensor & weight);
static inline std::tuple<Tensor &,Tensor &> prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight);
static inline std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask={{true, true}});
static inline Tensor & rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator * generator=nullptr);
static inline Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator * generator=nullptr);
static inline Tensor & rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
static inline Tensor rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
static inline Tensor & rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training);
static inline Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training);
static inline Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator * generator=nullptr);
static inline Tensor & rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
static inline Tensor & softmax_out(Tensor & output, const Tensor & self, int64_t dim);
static inline Tensor softmax(const Tensor & self, int64_t dim);
static inline Tensor & softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim);
static inline Tensor softmax_forward(const Tensor & self, int64_t dim);
static inline Tensor & softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output);
static inline Tensor softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output);
static inline Tensor & softplus_out(Tensor & output, const Tensor & self, Scalar beta=1, Scalar threshold=20);
static inline Tensor softplus(const Tensor & self, Scalar beta=1, Scalar threshold=20);
static inline Tensor & softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold);
static inline Tensor softplus_forward(const Tensor & self, Scalar beta, Scalar threshold);
static inline Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output);
static inline Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output);
static inline Tensor & softshrink_out(Tensor & output, const Tensor & self, Scalar lambd=0.5);
static inline Tensor softshrink(const Tensor & self, Scalar lambd=0.5);
static inline Tensor & softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd);
static inline Tensor softshrink_forward(const Tensor & self, Scalar lambd);
static inline Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd);
static inline Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd);
static inline Tensor & threshold_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor threshold(const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor & threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor threshold_forward(const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor & threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value);
static inline Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value);
static inline Tensor & threshold_forward_(Tensor & self, Scalar threshold, Scalar value);
static inline Tensor & adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor adaptive_avg_pool2d(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self);
static inline Tensor adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self);
static inline Tensor & adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor adaptive_avg_pool3d(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self);
static inline Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntList output_size);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d_forward(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices);
static inline Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntList output_size);
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool3d_forward(const Tensor & self, IntList output_size);
static inline Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices);
static inline Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices);
static inline Tensor & avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor & avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, bool ceil_mode=false, bool count_include_pad=false);
static inline Tensor & avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad);
static inline std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples);
static inline std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples);
static inline std::tuple<Tensor &,Tensor &> fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples);
static inline std::tuple<Tensor,Tensor> fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples);
static inline Tensor & fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices);
static inline Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices);
static inline std::tuple<Tensor &,Tensor &> max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor,Tensor> max_pool2d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor &,Tensor &> max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline std::tuple<Tensor,Tensor> max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline Tensor & max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline std::tuple<Tensor &,Tensor &> max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor,Tensor> max_pool3d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline std::tuple<Tensor &,Tensor &> max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline std::tuple<Tensor,Tensor> max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode);
static inline Tensor & max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices);
static inline Tensor & max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size);
static inline Tensor & max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding);
static inline Tensor & reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad1d(const Tensor & self, IntList padding);
static inline Tensor & reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad1d_forward(const Tensor & self, IntList padding);
static inline Tensor & reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor & reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad2d(const Tensor & self, IntList padding);
static inline Tensor & reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad2d_forward(const Tensor & self, IntList padding);
static inline Tensor & reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor & replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad1d(const Tensor & self, IntList padding);
static inline Tensor & replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad1d_forward(const Tensor & self, IntList padding);
static inline Tensor & replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor & replication_pad2d_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad2d(const Tensor & self, IntList padding);
static inline Tensor & replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad2d_forward(const Tensor & self, IntList padding);
static inline Tensor & replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor & replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad3d(const Tensor & self, IntList padding);
static inline Tensor & replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding);
static inline Tensor replication_pad3d_forward(const Tensor & self, IntList padding);
static inline Tensor & replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding);
static inline Tensor & upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_linear1d(const Tensor & self, IntList output_size);
static inline Tensor & upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_linear1d_forward(const Tensor & self, IntList output_size);
static inline Tensor & upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor & upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_bilinear2d(const Tensor & self, IntList output_size);
static inline Tensor & upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_bilinear2d_forward(const Tensor & self, IntList output_size);
static inline Tensor & upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor & upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_trilinear3d(const Tensor & self, IntList output_size);
static inline Tensor & upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size);
static inline Tensor upsample_trilinear3d_forward(const Tensor & self, IntList output_size);
static inline Tensor & upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size);
static inline Tensor & upsample_nearest1d_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest1d(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest2d_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest2d(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest3d_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest3d(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor);
static inline Tensor & upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor);
static inline Tensor & _sigmoid_out(Tensor & output, const Tensor & self);
static inline Tensor _sigmoid(const Tensor & self);
static inline Tensor & _sigmoid_forward_out(Tensor & output, const Tensor & self);
static inline Tensor _sigmoid_forward(const Tensor & self);
static inline Tensor & _sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
static inline Tensor _sigmoid_backward(const Tensor & grad_output, const Tensor & output);
static inline Tensor & _tanh_out(Tensor & output, const Tensor & self);
static inline Tensor _tanh(const Tensor & self);
static inline Tensor & _tanh_forward_out(Tensor & output, const Tensor & self);
static inline Tensor _tanh_forward(const Tensor & self);
static inline Tensor & _tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
static inline Tensor _tanh_backward(const Tensor & grad_output, const Tensor & output);
static inline Tensor & thnn_batch_norm_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline Tensor thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline Tensor thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, IntList dilation=1);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor & thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask={{true, true}});
static inline Tensor & thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline Tensor thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor & thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline Tensor thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation);
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones);
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask={{true, true, true}});
static inline Tensor adaptive_avg_pool1d(const Tensor & self, IntList output_size);
static inline std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntList output_size);
static inline bool allclose(const Tensor & self, const Tensor & other, double rtol=1e-05, double atol=1e-08);
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta=1, Scalar alpha=1);
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta=1, Scalar alpha=1);
static inline Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled);
static inline Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator=nullptr);
static inline Tensor & bernoulli_(Tensor & self, double p=0.5, Generator * generator=nullptr);
static inline Tensor cat(TensorList tensors, int64_t dim=0);
static inline Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim=0);
static inline Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline Tensor & sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta=1, Scalar alpha=1);
static inline std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim=0);
static inline bool cudnn_is_acceptable(const Tensor & self);
static inline Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups);
static inline Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled);
static inline Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding);
static inline std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask);
static inline Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
static inline Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
static inline Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList dilation=1, int64_t groups=1);
static inline Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad);
static inline std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad);
static inline Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
static inline Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
static inline Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias={}, IntList stride=1, IntList padding=0, IntList output_padding=0, int64_t groups=1, IntList dilation=1);
static inline Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W);
static inline Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W);
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon);
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon);
static inline Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline Tensor cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask);
static inline Tensor cudnn_convolution_backward_bias(const Tensor & grad_output);
static inline Tensor cudnn_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask);
static inline Tensor cudnn_convolution_transpose_backward_bias(const Tensor & grad_output);
static inline Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline Tensor cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic);
static inline Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid);
static inline std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output);
static inline Tensor det(const Tensor & self);
static inline std::tuple<Tensor,Tensor,Tensor,Tensor> _det_with_svd(const Tensor & self);
static inline Tensor dot(const Tensor & self, const Tensor & tensor);
static inline Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx=-1, bool scale_grad_by_freq=false, bool sparse=false);
static inline Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse);
static inline Tensor embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
static inline Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type);
static inline Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);
static inline Tensor empty_like(const Tensor & self);
static inline std::tuple<Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq=false, int64_t mode=0, bool sparse=false);
static inline Tensor embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse);
static inline Tensor embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode);
static inline Tensor embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode);
static inline Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, bool size_average, bool reduce);
static inline Tensor ger(const Tensor & self, const Tensor & vec2);
static inline Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2);
static inline Tensor index(const Tensor & self, TensorList indices);
static inline Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values);
static inline bool is_cuda(const Tensor & self);
static inline bool is_distributed(const Tensor & self);
static inline bool is_floating_point(const Tensor & self);
static inline bool is_nonzero(const Tensor & self);
static inline bool is_same_size(const Tensor & self, const Tensor & other);
static inline bool is_signed(const Tensor & self);
static inline bool is_sparse(const Tensor & self);
static inline Tensor matmul(const Tensor & self, const Tensor & other);
static inline std::tuple<Tensor,Tensor> max_pool1d(const Tensor & self, IntList kernel_size, IntList stride={}, IntList padding=0, IntList dilation=1, bool ceil_mode=false);
static inline Tensor mm(const Tensor & self, const Tensor & mat2);
static inline Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
static inline Tensor mv(const Tensor & self, const Tensor & vec);
static inline Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec);
static inline Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length);
static inline Tensor pin_memory(const Tensor & self);
static inline Tensor rand_like(const Tensor & self);
static inline Tensor randn_like(const Tensor & self);
static inline Tensor repeat(const Tensor & self, IntList repeats);
static inline std::tuple<Tensor,Tensor> RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale);
static inline Tensor RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes);
static inline Tensor rrelu(const Tensor & self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator * generator=nullptr);
static inline Tensor & rrelu_(Tensor & self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=false, Generator * generator=nullptr);
static inline Tensor select(const Tensor & self, int64_t dim, int64_t index);
static inline Tensor selu(const Tensor & self);
static inline Tensor & selu_(Tensor & self);
static inline int64_t size(const Tensor & self, int64_t dim);
static inline Tensor slice(const Tensor & self, int64_t dim=0, int64_t start=0, int64_t end=9223372036854775807, int64_t step=1);
static inline std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim=0);
static inline Tensor squeeze(const Tensor & self);
static inline Tensor squeeze(const Tensor & self, int64_t dim);
static inline Tensor & squeeze_(Tensor & self);
static inline Tensor & squeeze_(Tensor & self, int64_t dim);
static inline Tensor stack(TensorList tensors, int64_t dim=0);
static inline Tensor & stack_out(Tensor & result, TensorList tensors, int64_t dim=0);
static inline Tensor stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool return_onesided=true, const Tensor & window={}, int64_t pad_end=0);
static inline int64_t stride(const Tensor & self, int64_t dim);
static inline Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1);
static inline Tensor & t_(Tensor & self);
static inline Tensor type_as(const Tensor & self, const Tensor & other);
static inline Tensor unsqueeze(const Tensor & self, int64_t dim);
static inline Tensor & unsqueeze_(Tensor & self, int64_t dim);
static inline Tensor view_as(const Tensor & self, const Tensor & other);
static inline Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other);
static inline Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other);
static inline Tensor _standard_gamma_grad(const Tensor & self, const Tensor & output);
static inline Tensor poisson(const Tensor & self, Generator * generator=nullptr);
static inline Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional);
static inline std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state);
static inline std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask);

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
static inline Tensor & take_out(Tensor & result, const Tensor & self, const Tensor & index) {
    return infer_type(self).take_out(result, self, index);
}
static inline Tensor take(const Tensor & self, const Tensor & index) {
    return infer_type(self).take(self, index);
}
static inline Tensor & range_out(Tensor & result, Scalar start, Scalar end, Scalar step) {
    return infer_type(result).range_out(result, start, end, step);
}
static inline Tensor & arange_out(Tensor & result, Scalar start, Scalar end, Scalar step) {
    return infer_type(result).arange_out(result, start, end, step);
}
static inline Tensor & arange_out(Tensor & result, Scalar end) {
    return infer_type(result).arange_out(result, end);
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
static inline Tensor min(const Tensor & self) {
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
static inline Tensor max(const Tensor & self) {
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
static inline Tensor median(const Tensor & self) {
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
static inline Tensor & abs_out(Tensor & result, const Tensor & self) {
    return infer_type(self).abs_out(result, self);
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
static inline Tensor & digamma_out(Tensor & result, const Tensor & self) {
    return infer_type(self).digamma_out(result, self);
}
static inline Tensor digamma(const Tensor & self) {
    return infer_type(self).digamma(self);
}
static inline Tensor & polygamma_out(Tensor & result, int64_t n, const Tensor & self) {
    return infer_type(self).polygamma_out(result, n, self);
}
static inline Tensor polygamma(int64_t n, const Tensor & self) {
    return infer_type(self).polygamma(n, self);
}
static inline Tensor & exp_out(Tensor & result, const Tensor & self) {
    return infer_type(self).exp_out(result, self);
}
static inline Tensor exp(const Tensor & self) {
    return infer_type(self).exp(self);
}
static inline Tensor & expm1_out(Tensor & result, const Tensor & self) {
    return infer_type(self).expm1_out(result, self);
}
static inline Tensor expm1(const Tensor & self) {
    return infer_type(self).expm1(self);
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
static inline Tensor & mean_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mean_out(result, self, dim, keepdim);
}
static inline Tensor mean(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).mean(self, dim, keepdim);
}
static inline Tensor mean(const Tensor & self) {
    return infer_type(self).mean(self);
}
static inline Tensor & var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).var_out(result, self, dim, unbiased, keepdim);
}
static inline Tensor var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).var(self, dim, unbiased, keepdim);
}
static inline Tensor var(const Tensor & self, bool unbiased) {
    return infer_type(self).var(self, unbiased);
}
static inline Tensor & std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).std_out(result, self, dim, unbiased, keepdim);
}
static inline Tensor std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim) {
    return infer_type(self).std(self, dim, unbiased, keepdim);
}
static inline Tensor std(const Tensor & self, bool unbiased) {
    return infer_type(self).std(self, unbiased);
}
static inline Tensor & norm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    return infer_type(self).norm_out(result, self, p, dim, keepdim);
}
static inline Tensor norm(const Tensor & self, Scalar p, int64_t dim, bool keepdim) {
    return infer_type(self).norm(self, p, dim, keepdim);
}
static inline Tensor norm(const Tensor & self, Scalar p) {
    return infer_type(self).norm(self, p);
}
static inline Tensor & renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    return infer_type(self).renorm_out(result, self, p, dim, maxnorm);
}
static inline Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm) {
    return infer_type(self).renorm(self, p, dim, maxnorm);
}
static inline Tensor dist(const Tensor & self, const Tensor & other, Scalar p) {
    return infer_type(self).dist(self, other, p);
}
static inline Tensor & reciprocal_out(Tensor & result, const Tensor & self) {
    return infer_type(self).reciprocal_out(result, self);
}
static inline Tensor reciprocal(const Tensor & self) {
    return infer_type(self).reciprocal(self);
}
static inline Tensor & neg_out(Tensor & result, const Tensor & self) {
    return infer_type(self).neg_out(result, self);
}
static inline Tensor neg(const Tensor & self) {
    return infer_type(self).neg(self);
}
static inline Tensor & atan2_out(Tensor & result, const Tensor & self, const Tensor & other) {
    return infer_type(self).atan2_out(result, self, other);
}
static inline Tensor atan2(const Tensor & self, const Tensor & other) {
    return infer_type(self).atan2(self, other);
}
static inline Tensor & pow_out(Tensor & result, const Tensor & self, Scalar exponent) {
    return infer_type(self).pow_out(result, self, exponent);
}
static inline Tensor pow(const Tensor & self, Scalar exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & pow_out(Tensor & result, const Tensor & self, const Tensor & exponent) {
    return infer_type(self).pow_out(result, self, exponent);
}
static inline Tensor pow(const Tensor & self, const Tensor & exponent) {
    return infer_type(self).pow(self, exponent);
}
static inline Tensor & pow_out(Tensor & result, Scalar base, const Tensor & self) {
    return infer_type(self).pow_out(result, base, self);
}
static inline Tensor pow(Scalar base, const Tensor & self) {
    return infer_type(self).pow(base, self);
}
static inline Tensor & lerp_out(Tensor & result, const Tensor & self, const Tensor & end, Scalar weight) {
    return infer_type(self).lerp_out(result, self, end, weight);
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
static inline Tensor & histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max) {
    return infer_type(self).histc_out(result, self, bins, min, max);
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
static inline Tensor sum(const Tensor & self) {
    return infer_type(self).sum(self);
}
static inline Tensor & prod_out(Tensor & result, const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).prod_out(result, self, dim, keepdim);
}
static inline Tensor prod(const Tensor & self, int64_t dim, bool keepdim) {
    return infer_type(self).prod(self, dim, keepdim);
}
static inline Tensor prod(const Tensor & self) {
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
static inline Tensor trace(const Tensor & self) {
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
static inline Tensor & clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp_out(result, self, min, max);
}
static inline Tensor clamp(const Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp(self, min, max);
}
static inline Tensor & clamp_(Tensor & self, Scalar min, Scalar max) {
    return infer_type(self).clamp_(self, min, max);
}
static inline Tensor & clamp_min_out(Tensor & result, const Tensor & self, Scalar min) {
    return infer_type(self).clamp_min_out(result, self, min);
}
static inline Tensor clamp_min(const Tensor & self, Scalar min) {
    return infer_type(self).clamp_min(self, min);
}
static inline Tensor & clamp_min_(Tensor & self, Scalar min) {
    return infer_type(self).clamp_min_(self, min);
}
static inline Tensor & clamp_max_out(Tensor & result, const Tensor & self, Scalar max) {
    return infer_type(self).clamp_max_out(result, self, max);
}
static inline Tensor clamp_max(const Tensor & self, Scalar max) {
    return infer_type(self).clamp_max(self, max);
}
static inline Tensor & clamp_max_(Tensor & self, Scalar max) {
    return infer_type(self).clamp_max_(self, max);
}
static inline Tensor _dot(const Tensor & self, const Tensor & tensor) {
    return infer_type(self)._dot(self, tensor);
}
static inline Tensor & tril_out(Tensor & result, const Tensor & self, int64_t diagonal) {
    return infer_type(self).tril_out(result, self, diagonal);
}
static inline Tensor tril(const Tensor & self, int64_t diagonal) {
    return infer_type(self).tril(self, diagonal);
}
static inline Tensor & triu_out(Tensor & result, const Tensor & self, int64_t diagonal) {
    return infer_type(self).triu_out(result, self, diagonal);
}
static inline Tensor triu(const Tensor & self, int64_t diagonal) {
    return infer_type(self).triu(self, diagonal);
}
static inline Tensor & cross_out(Tensor & result, const Tensor & self, const Tensor & other, int64_t dim) {
    return infer_type(self).cross_out(result, self, other, dim);
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
static inline Tensor & addmm_out(Tensor & result, const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).addmm_out(result, self, mat1, mat2, beta, alpha);
}
static inline Tensor addmm(const Tensor & self, SparseTensor mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).addmm(self, mat1, mat2, beta, alpha);
}
static inline Tensor & _addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self)._addmv_out(result, self, mat, vec, beta, alpha);
}
static inline Tensor _addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self)._addmv(self, mat, vec, beta, alpha);
}
static inline Tensor & _addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self)._addr_out(result, self, vec1, vec2, beta, alpha);
}
static inline Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self)._addr(self, vec1, vec2, beta, alpha);
}
static inline Tensor & _ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) {
    return infer_type(self)._ger_out(result, self, vec2);
}
static inline Tensor _ger(const Tensor & self, const Tensor & vec2) {
    return infer_type(self)._ger(self, vec2);
}
static inline Tensor & _mv_out(Tensor & result, const Tensor & self, const Tensor & vec) {
    return infer_type(self)._mv_out(result, self, vec);
}
static inline Tensor _mv(const Tensor & self, const Tensor & vec) {
    return infer_type(self)._mv(self, vec);
}
static inline Tensor & _mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) {
    return infer_type(self)._mm_out(result, self, mat2);
}
static inline Tensor _mm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self)._mm(self, mat2);
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
static inline std::tuple<Tensor &,Tensor &> btrifact_out(Tensor & result, Tensor & pivots, const Tensor & self, bool pivot) {
    return infer_type(self).btrifact_out(result, pivots, self, pivot);
}
static inline std::tuple<Tensor,Tensor> btrifact(const Tensor & self, bool pivot) {
    return infer_type(self).btrifact(self, pivot);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> btrifact_with_info_out(Tensor & result, Tensor & pivots, Tensor & info, const Tensor & self, bool pivot) {
    return infer_type(self).btrifact_with_info_out(result, pivots, info, self, pivot);
}
static inline std::tuple<Tensor,Tensor,Tensor> btrifact_with_info(const Tensor & self, bool pivot) {
    return infer_type(self).btrifact_with_info(self, pivot);
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
static inline Tensor & normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator) {
    return infer_type(output).normal_out(output, mean, std, generator);
}
static inline Tensor normal(const Tensor & mean, double std, Generator * generator) {
    return infer_type(mean).normal(mean, std, generator);
}
static inline Tensor & normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator) {
    return infer_type(output).normal_out(output, mean, std, generator);
}
static inline Tensor normal(double mean, const Tensor & std, Generator * generator) {
    return infer_type(std).normal(mean, std, generator);
}
static inline Tensor & normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator) {
    return infer_type(output).normal_out(output, mean, std, generator);
}
static inline Tensor normal(const Tensor & mean, const Tensor & std, Generator * generator) {
    return infer_type(mean).normal(mean, std, generator);
}
static inline Tensor & rand_out(Tensor & result, IntList size, Generator * generator) {
    return infer_type(result).rand_out(result, size, generator);
}
static inline Tensor & randn_out(Tensor & result, IntList size, Generator * generator) {
    return infer_type(result).randn_out(result, size, generator);
}
static inline Tensor & bernoulli_out(Tensor & output, const Tensor & self, Generator * generator) {
    return infer_type(self).bernoulli_out(output, self, generator);
}
static inline Tensor bernoulli(const Tensor & self, Generator * generator) {
    return infer_type(self).bernoulli(self, generator);
}
static inline Tensor & _standard_gamma_out(Tensor & output, const Tensor & self, Generator * generator) {
    return infer_type(self)._standard_gamma_out(output, self, generator);
}
static inline Tensor _standard_gamma(const Tensor & self, Generator * generator) {
    return infer_type(self)._standard_gamma(self, generator);
}
static inline Tensor & _dirichlet_grad_out(Tensor & output, const Tensor & x, const Tensor & alpha, const Tensor & total) {
    return infer_type(output)._dirichlet_grad_out(output, x, alpha, total);
}
static inline Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total) {
    return infer_type(x)._dirichlet_grad(x, alpha, total);
}
static inline Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntList size) {
    return infer_type(values).sparse_coo_tensor(indices, values, size);
}
static inline Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values) {
    return infer_type(values).sparse_coo_tensor(indices, values);
}
static inline Tensor alias(const Tensor & self) {
    return infer_type(self).alias(self);
}
static inline Tensor & as_strided_out(Tensor & result, const Tensor & self, IntList size, IntList stride, int64_t storage_offset) {
    return infer_type(self).as_strided_out(result, self, size, stride, storage_offset);
}
static inline Tensor as_strided(const Tensor & self, IntList size, IntList stride, int64_t storage_offset) {
    return infer_type(self).as_strided(self, size, stride, storage_offset);
}
static inline Tensor & as_strided_(Tensor & self, IntList size, IntList stride, int64_t storage_offset) {
    return infer_type(self).as_strided_(self, size, stride, storage_offset);
}
static inline Tensor & _cat_out(Tensor & self, TensorList tensors, int64_t dim) {
    return infer_type(self)._cat_out(self, tensors, dim);
}
static inline Tensor _cat(TensorList tensors, int64_t dim) {
    return infer_type(tensors)._cat(tensors, dim);
}
static inline Tensor & binary_cross_entropy_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy_out(output, self, target, weight, size_average, reduce);
}
static inline Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy(self, target, weight, size_average, reduce);
}
static inline Tensor & binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy_forward_out(output, self, target, weight, size_average, reduce);
}
static inline Tensor binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy_forward(self, target, weight, size_average, reduce);
}
static inline Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy_backward_out(grad_input, grad_output, self, target, weight, size_average, reduce);
}
static inline Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, bool reduce) {
    return infer_type(self).binary_cross_entropy_backward(grad_output, self, target, weight, size_average, reduce);
}
static inline Tensor & kl_div_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div_out(output, self, target, size_average, reduce);
}
static inline Tensor kl_div(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div(self, target, size_average, reduce);
}
static inline Tensor & kl_div_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div_forward_out(output, self, target, size_average, reduce);
}
static inline Tensor kl_div_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div_forward(self, target, size_average, reduce);
}
static inline Tensor & kl_div_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div_backward_out(grad_input, grad_output, self, target, size_average, reduce);
}
static inline Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).kl_div_backward(grad_output, self, target, size_average, reduce);
}
static inline Tensor & l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss_out(output, self, target, size_average, reduce);
}
static inline Tensor l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss(self, target, size_average, reduce);
}
static inline Tensor & l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss_forward_out(output, self, target, size_average, reduce);
}
static inline Tensor l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss_forward(self, target, size_average, reduce);
}
static inline Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss_backward_out(grad_input, grad_output, self, target, size_average, reduce);
}
static inline Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).l1_loss_backward(grad_output, self, target, size_average, reduce);
}
static inline Tensor & mse_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss_out(output, self, target, size_average, reduce);
}
static inline Tensor mse_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss(self, target, size_average, reduce);
}
static inline Tensor & mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss_forward_out(output, self, target, size_average, reduce);
}
static inline Tensor mse_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss_forward(self, target, size_average, reduce);
}
static inline Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss_backward_out(grad_input, grad_output, self, target, size_average, reduce);
}
static inline Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).mse_loss_backward(grad_output, self, target, size_average, reduce);
}
static inline Tensor & multi_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss_out(output, self, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss(self, target, p, margin, weight, size_average);
}
static inline Tensor & multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss_forward_out(output, self, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss_forward(self, target, p, margin, weight, size_average);
}
static inline Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss_backward_out(grad_input, self, target, p, margin, weight, size_average);
}
static inline Tensor multi_margin_loss_backward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, bool size_average) {
    return infer_type(self).multi_margin_loss_backward(self, target, p, margin, weight, size_average);
}
static inline Tensor & multilabel_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).multilabel_margin_loss_out(output, self, target, size_average, reduce);
}
static inline Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).multilabel_margin_loss(self, target, size_average, reduce);
}
static inline std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).multilabel_margin_loss_forward_out(output, is_target, self, target, size_average, reduce);
}
static inline std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).multilabel_margin_loss_forward(self, target, size_average, reduce);
}
static inline Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) {
    return infer_type(self).multilabel_margin_loss_backward_out(grad_input, grad_output, self, target, size_average, reduce, is_target);
}
static inline Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce, const Tensor & is_target) {
    return infer_type(self).multilabel_margin_loss_backward(grad_output, self, target, size_average, reduce, is_target);
}
static inline Tensor & nll_loss_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss_out(output, self, target, weight, size_average, ignore_index, reduce);
}
static inline Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss(self, target, weight, size_average, ignore_index, reduce);
}
static inline std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss_forward_out(output, total_weight, self, target, weight, size_average, ignore_index, reduce);
}
static inline std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss_forward(self, target, weight, size_average, ignore_index, reduce);
}
static inline Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) {
    return infer_type(self).nll_loss_backward_out(grad_input, grad_output, self, target, weight, size_average, ignore_index, reduce, total_weight);
}
static inline Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) {
    return infer_type(self).nll_loss_backward(grad_output, self, target, weight, size_average, ignore_index, reduce, total_weight);
}
static inline Tensor & nll_loss2d_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss2d_out(output, self, target, weight, size_average, ignore_index, reduce);
}
static inline Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss2d(self, target, weight, size_average, ignore_index, reduce);
}
static inline std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss2d_forward_out(output, total_weight, self, target, weight, size_average, ignore_index, reduce);
}
static inline std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce) {
    return infer_type(self).nll_loss2d_forward(self, target, weight, size_average, ignore_index, reduce);
}
static inline Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) {
    return infer_type(self).nll_loss2d_backward_out(grad_input, grad_output, self, target, weight, size_average, ignore_index, reduce, total_weight);
}
static inline Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, bool size_average, int64_t ignore_index, bool reduce, const Tensor & total_weight) {
    return infer_type(self).nll_loss2d_backward(grad_output, self, target, weight, size_average, ignore_index, reduce, total_weight);
}
static inline Tensor & smooth_l1_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss_out(output, self, target, size_average, reduce);
}
static inline Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss(self, target, size_average, reduce);
}
static inline Tensor & smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss_forward_out(output, self, target, size_average, reduce);
}
static inline Tensor smooth_l1_loss_forward(const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss_forward(self, target, size_average, reduce);
}
static inline Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss_backward_out(grad_input, grad_output, self, target, size_average, reduce);
}
static inline Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, bool size_average, bool reduce) {
    return infer_type(self).smooth_l1_loss_backward(grad_output, self, target, size_average, reduce);
}
static inline Tensor & soft_margin_loss_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss_out(output, self, target, size_average);
}
static inline Tensor soft_margin_loss(const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss(self, target, size_average);
}
static inline Tensor & soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss_forward_out(output, self, target, size_average);
}
static inline Tensor soft_margin_loss_forward(const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss_forward(self, target, size_average);
}
static inline Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss_backward_out(grad_input, self, target, size_average);
}
static inline Tensor soft_margin_loss_backward(const Tensor & self, const Tensor & target, bool size_average) {
    return infer_type(self).soft_margin_loss_backward(self, target, size_average);
}
static inline Tensor & elu_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu_out(output, self, alpha, scale);
}
static inline Tensor elu(const Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu(self, alpha, scale);
}
static inline Tensor & elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu_forward_out(output, self, alpha, scale);
}
static inline Tensor elu_forward(const Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu_forward(self, alpha, scale);
}
static inline Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) {
    return infer_type(grad_input).elu_backward_out(grad_input, grad_output, alpha, scale, output);
}
static inline Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, const Tensor & output) {
    return infer_type(grad_output).elu_backward(grad_output, alpha, scale, output);
}
static inline Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu_(self, alpha, scale);
}
static inline Tensor & elu_forward_(Tensor & self, Scalar alpha, Scalar scale) {
    return infer_type(self).elu_forward_(self, alpha, scale);
}
static inline Tensor & glu_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).glu_out(output, self, dim);
}
static inline Tensor glu(const Tensor & self, int64_t dim) {
    return infer_type(self).glu(self, dim);
}
static inline Tensor & glu_forward_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).glu_forward_out(output, self, dim);
}
static inline Tensor glu_forward(const Tensor & self, int64_t dim) {
    return infer_type(self).glu_forward(self, dim);
}
static inline Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim) {
    return infer_type(self).glu_backward_out(grad_input, grad_output, self, dim);
}
static inline Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim) {
    return infer_type(self).glu_backward(grad_output, self, dim);
}
static inline Tensor & hardshrink_out(Tensor & output, const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink_out(output, self, lambd);
}
static inline Tensor hardshrink(const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink(self, lambd);
}
static inline Tensor & hardshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink_forward_out(output, self, lambd);
}
static inline Tensor hardshrink_forward(const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink_forward(self, lambd);
}
static inline Tensor & hardshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink_backward_out(grad_input, grad_output, self, lambd);
}
static inline Tensor hardshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
    return infer_type(self).hardshrink_backward(grad_output, self, lambd);
}
static inline Tensor & hardtanh_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_out(output, self, min_val, max_val);
}
static inline Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh(self, min_val, max_val);
}
static inline Tensor & hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_forward_out(output, self, min_val, max_val);
}
static inline Tensor hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_forward(self, min_val, max_val);
}
static inline Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_backward_out(grad_input, grad_output, self, min_val, max_val);
}
static inline Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_backward(grad_output, self, min_val, max_val);
}
static inline Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_(self, min_val, max_val);
}
static inline Tensor & hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val) {
    return infer_type(self).hardtanh_forward_(self, min_val, max_val);
}
static inline Tensor & leaky_relu_out(Tensor & output, const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_out(output, self, negative_slope);
}
static inline Tensor leaky_relu(const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu(self, negative_slope);
}
static inline Tensor & leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_forward_out(output, self, negative_slope);
}
static inline Tensor leaky_relu_forward(const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_forward(self, negative_slope);
}
static inline Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_backward_out(grad_input, grad_output, self, negative_slope);
}
static inline Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_backward(grad_output, self, negative_slope);
}
static inline Tensor & leaky_relu_(Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_(self, negative_slope);
}
static inline Tensor & leaky_relu_forward_(Tensor & self, Scalar negative_slope) {
    return infer_type(self).leaky_relu_forward_(self, negative_slope);
}
static inline Tensor & log_sigmoid_out(Tensor & output, const Tensor & self) {
    return infer_type(self).log_sigmoid_out(output, self);
}
static inline Tensor log_sigmoid(const Tensor & self) {
    return infer_type(self).log_sigmoid(self);
}
static inline std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self) {
    return infer_type(self).log_sigmoid_forward_out(output, buffer, self);
}
static inline std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self) {
    return infer_type(self).log_sigmoid_forward(self);
}
static inline Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
    return infer_type(self).log_sigmoid_backward_out(grad_input, grad_output, self, buffer);
}
static inline Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer) {
    return infer_type(self).log_sigmoid_backward(grad_output, self, buffer);
}
static inline Tensor & log_softmax_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).log_softmax_out(output, self, dim);
}
static inline Tensor log_softmax(const Tensor & self, int64_t dim) {
    return infer_type(self).log_softmax(self, dim);
}
static inline Tensor & log_softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).log_softmax_forward_out(output, self, dim);
}
static inline Tensor log_softmax_forward(const Tensor & self, int64_t dim) {
    return infer_type(self).log_softmax_forward(self, dim);
}
static inline Tensor & log_softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) {
    return infer_type(self).log_softmax_backward_out(grad_input, grad_output, self, dim, output);
}
static inline Tensor log_softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) {
    return infer_type(self).log_softmax_backward(grad_output, self, dim, output);
}
static inline Tensor & prelu_out(Tensor & output, const Tensor & self, const Tensor & weight) {
    return infer_type(self).prelu_out(output, self, weight);
}
static inline Tensor prelu(const Tensor & self, const Tensor & weight) {
    return infer_type(self).prelu(self, weight);
}
static inline Tensor & prelu_forward_out(Tensor & output, const Tensor & self, const Tensor & weight) {
    return infer_type(self).prelu_forward_out(output, self, weight);
}
static inline Tensor prelu_forward(const Tensor & self, const Tensor & weight) {
    return infer_type(self).prelu_forward(self, weight);
}
static inline std::tuple<Tensor &,Tensor &> prelu_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight) {
    return infer_type(self).prelu_backward_out(grad_input, grad_weight, grad_output, self, weight);
}
static inline std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, std::array<bool,2> output_mask) {
    return infer_type(self).prelu_backward(grad_output, self, weight, output_mask);
}
static inline Tensor & rrelu_with_noise_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise_out(output, self, noise, lower, upper, training, generator);
}
static inline Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise(self, noise, lower, upper, training, generator);
}
static inline Tensor & rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise_forward_out(output, self, noise, lower, upper, training, generator);
}
static inline Tensor rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise_forward(self, noise, lower, upper, training, generator);
}
static inline Tensor & rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
    return infer_type(self).rrelu_with_noise_backward_out(grad_input, grad_output, self, noise, lower, upper, training);
}
static inline Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training) {
    return infer_type(self).rrelu_with_noise_backward(grad_output, self, noise, lower, upper, training);
}
static inline Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise_(self, noise, lower, upper, training, generator);
}
static inline Tensor & rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_with_noise_forward_(self, noise, lower, upper, training, generator);
}
static inline Tensor & softmax_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).softmax_out(output, self, dim);
}
static inline Tensor softmax(const Tensor & self, int64_t dim) {
    return infer_type(self).softmax(self, dim);
}
static inline Tensor & softmax_forward_out(Tensor & output, const Tensor & self, int64_t dim) {
    return infer_type(self).softmax_forward_out(output, self, dim);
}
static inline Tensor softmax_forward(const Tensor & self, int64_t dim) {
    return infer_type(self).softmax_forward(self, dim);
}
static inline Tensor & softmax_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) {
    return infer_type(self).softmax_backward_out(grad_input, grad_output, self, dim, output);
}
static inline Tensor softmax_backward(const Tensor & grad_output, const Tensor & self, int64_t dim, const Tensor & output) {
    return infer_type(self).softmax_backward(grad_output, self, dim, output);
}
static inline Tensor & softplus_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) {
    return infer_type(self).softplus_out(output, self, beta, threshold);
}
static inline Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold) {
    return infer_type(self).softplus(self, beta, threshold);
}
static inline Tensor & softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold) {
    return infer_type(self).softplus_forward_out(output, self, beta, threshold);
}
static inline Tensor softplus_forward(const Tensor & self, Scalar beta, Scalar threshold) {
    return infer_type(self).softplus_forward(self, beta, threshold);
}
static inline Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
    return infer_type(self).softplus_backward_out(grad_input, grad_output, self, beta, threshold, output);
}
static inline Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output) {
    return infer_type(self).softplus_backward(grad_output, self, beta, threshold, output);
}
static inline Tensor & softshrink_out(Tensor & output, const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink_out(output, self, lambd);
}
static inline Tensor softshrink(const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink(self, lambd);
}
static inline Tensor & softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink_forward_out(output, self, lambd);
}
static inline Tensor softshrink_forward(const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink_forward(self, lambd);
}
static inline Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink_backward_out(grad_input, grad_output, self, lambd);
}
static inline Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd) {
    return infer_type(self).softshrink_backward(grad_output, self, lambd);
}
static inline Tensor & threshold_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_out(output, self, threshold, value);
}
static inline Tensor threshold(const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold(self, threshold, value);
}
static inline Tensor & threshold_forward_out(Tensor & output, const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_forward_out(output, self, threshold, value);
}
static inline Tensor threshold_forward(const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_forward(self, threshold, value);
}
static inline Tensor & threshold_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_backward_out(grad_input, grad_output, self, threshold, value);
}
static inline Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_backward(grad_output, self, threshold, value);
}
static inline Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_(self, threshold, value);
}
static inline Tensor & threshold_forward_(Tensor & self, Scalar threshold, Scalar value) {
    return infer_type(self).threshold_forward_(self, threshold, value);
}
static inline Tensor & adaptive_avg_pool2d_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool2d_out(output, self, output_size);
}
static inline Tensor adaptive_avg_pool2d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool2d(self, output_size);
}
static inline Tensor & adaptive_avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool2d_forward_out(output, self, output_size);
}
static inline Tensor adaptive_avg_pool2d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool2d_forward(self, output_size);
}
static inline Tensor & adaptive_avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
    return infer_type(self).adaptive_avg_pool2d_backward_out(grad_input, grad_output, self);
}
static inline Tensor adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self) {
    return infer_type(self).adaptive_avg_pool2d_backward(grad_output, self);
}
static inline Tensor & adaptive_avg_pool3d_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool3d_out(output, self, output_size);
}
static inline Tensor adaptive_avg_pool3d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool3d(self, output_size);
}
static inline Tensor & adaptive_avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool3d_forward_out(output, self, output_size);
}
static inline Tensor adaptive_avg_pool3d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool3d_forward(self, output_size);
}
static inline Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self) {
    return infer_type(self).adaptive_avg_pool3d_backward_out(grad_input, grad_output, self);
}
static inline Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self) {
    return infer_type(self).adaptive_avg_pool3d_backward(grad_output, self);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool2d_out(output, indices, self, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool2d(self, output_size);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool2d_forward_out(output, indices, self, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool2d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool2d_forward(self, output_size);
}
static inline Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
    return infer_type(self).adaptive_max_pool2d_backward_out(grad_input, grad_output, self, indices);
}
static inline Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
    return infer_type(self).adaptive_max_pool2d_backward(grad_output, self, indices);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool3d_out(output, indices, self, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool3d(self, output_size);
}
static inline std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool3d_forward_out(output, indices, self, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool3d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool3d_forward(self, output_size);
}
static inline Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
    return infer_type(self).adaptive_max_pool3d_backward_out(grad_input, grad_output, self, indices);
}
static inline Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices) {
    return infer_type(self).adaptive_max_pool3d_backward(grad_output, self, indices);
}
static inline Tensor & avg_pool2d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool2d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool2d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_forward_out(Tensor & output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d_forward_out(output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d_forward(self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, bool ceil_mode, bool count_include_pad) {
    return infer_type(self).avg_pool3d_backward(grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad);
}
static inline std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
    return infer_type(self).fractional_max_pool2d_out(output, indices, self, kernel_size, output_size, random_samples);
}
static inline std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
    return infer_type(self).fractional_max_pool2d(self, kernel_size, output_size, random_samples);
}
static inline std::tuple<Tensor &,Tensor &> fractional_max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
    return infer_type(self).fractional_max_pool2d_forward_out(output, indices, self, kernel_size, output_size, random_samples);
}
static inline std::tuple<Tensor,Tensor> fractional_max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & random_samples) {
    return infer_type(self).fractional_max_pool2d_forward(self, kernel_size, output_size, random_samples);
}
static inline Tensor & fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) {
    return infer_type(self).fractional_max_pool2d_backward_out(grad_input, grad_output, self, kernel_size, output_size, indices);
}
static inline Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList output_size, const Tensor & indices) {
    return infer_type(self).fractional_max_pool2d_backward(grad_output, self, kernel_size, output_size, indices);
}
static inline std::tuple<Tensor &,Tensor &> max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool2d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool2d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool2d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor &,Tensor &> max_pool2d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool2d_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool2d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool2d_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline Tensor & max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(self).max_pool2d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(self).max_pool2d_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline std::tuple<Tensor &,Tensor &> max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool3d_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool3d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool3d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor &,Tensor &> max_pool3d_forward_out(Tensor & output, Tensor & indices, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool3d_forward_out(output, indices, self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline std::tuple<Tensor,Tensor> max_pool3d_forward(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool3d_forward(self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline Tensor & max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(self).max_pool3d_backward_out(grad_input, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode, const Tensor & indices) {
    return infer_type(self).max_pool3d_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices);
}
static inline Tensor & max_unpool2d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d_out(output, self, indices, output_size);
}
static inline Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d(self, indices, output_size);
}
static inline Tensor & max_unpool2d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d_forward_out(output, self, indices, output_size);
}
static inline Tensor max_unpool2d_forward(const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d_forward(self, indices, output_size);
}
static inline Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d_backward_out(grad_input, grad_output, self, indices, output_size);
}
static inline Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size) {
    return infer_type(self).max_unpool2d_backward(grad_output, self, indices, output_size);
}
static inline Tensor & max_unpool3d_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d_out(output, self, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d(self, indices, output_size, stride, padding);
}
static inline Tensor & max_unpool3d_forward_out(Tensor & output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d_forward_out(output, self, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d_forward(const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d_forward(self, indices, output_size, stride, padding);
}
static inline Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d_backward_out(grad_input, grad_output, self, indices, output_size, stride, padding);
}
static inline Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntList output_size, IntList stride, IntList padding) {
    return infer_type(self).max_unpool3d_backward(grad_output, self, indices, output_size, stride, padding);
}
static inline Tensor & reflection_pad1d_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d_out(output, self, padding);
}
static inline Tensor reflection_pad1d(const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d(self, padding);
}
static inline Tensor & reflection_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d_forward_out(output, self, padding);
}
static inline Tensor reflection_pad1d_forward(const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d_forward(self, padding);
}
static inline Tensor & reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d_backward_out(grad_input, grad_output, self, padding);
}
static inline Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad1d_backward(grad_output, self, padding);
}
static inline Tensor & reflection_pad2d_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d_out(output, self, padding);
}
static inline Tensor reflection_pad2d(const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d(self, padding);
}
static inline Tensor & reflection_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d_forward_out(output, self, padding);
}
static inline Tensor reflection_pad2d_forward(const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d_forward(self, padding);
}
static inline Tensor & reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d_backward_out(grad_input, grad_output, self, padding);
}
static inline Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).reflection_pad2d_backward(grad_output, self, padding);
}
static inline Tensor & replication_pad1d_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d_out(output, self, padding);
}
static inline Tensor replication_pad1d(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d(self, padding);
}
static inline Tensor & replication_pad1d_forward_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d_forward_out(output, self, padding);
}
static inline Tensor replication_pad1d_forward(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d_forward(self, padding);
}
static inline Tensor & replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d_backward_out(grad_input, grad_output, self, padding);
}
static inline Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad1d_backward(grad_output, self, padding);
}
static inline Tensor & replication_pad2d_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d_out(output, self, padding);
}
static inline Tensor replication_pad2d(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d(self, padding);
}
static inline Tensor & replication_pad2d_forward_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d_forward_out(output, self, padding);
}
static inline Tensor replication_pad2d_forward(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d_forward(self, padding);
}
static inline Tensor & replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d_backward_out(grad_input, grad_output, self, padding);
}
static inline Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad2d_backward(grad_output, self, padding);
}
static inline Tensor & replication_pad3d_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d_out(output, self, padding);
}
static inline Tensor replication_pad3d(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d(self, padding);
}
static inline Tensor & replication_pad3d_forward_out(Tensor & output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d_forward_out(output, self, padding);
}
static inline Tensor replication_pad3d_forward(const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d_forward(self, padding);
}
static inline Tensor & replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d_backward_out(grad_input, grad_output, self, padding);
}
static inline Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntList padding) {
    return infer_type(self).replication_pad3d_backward(grad_output, self, padding);
}
static inline Tensor & upsample_linear1d_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_linear1d_out(output, self, output_size);
}
static inline Tensor upsample_linear1d(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_linear1d(self, output_size);
}
static inline Tensor & upsample_linear1d_forward_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_linear1d_forward_out(output, self, output_size);
}
static inline Tensor upsample_linear1d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_linear1d_forward(self, output_size);
}
static inline Tensor & upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_input).upsample_linear1d_backward_out(grad_input, grad_output, output_size, input_size);
}
static inline Tensor upsample_linear1d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_output).upsample_linear1d_backward(grad_output, output_size, input_size);
}
static inline Tensor & upsample_bilinear2d_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_bilinear2d_out(output, self, output_size);
}
static inline Tensor upsample_bilinear2d(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_bilinear2d(self, output_size);
}
static inline Tensor & upsample_bilinear2d_forward_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_bilinear2d_forward_out(output, self, output_size);
}
static inline Tensor upsample_bilinear2d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_bilinear2d_forward(self, output_size);
}
static inline Tensor & upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_input).upsample_bilinear2d_backward_out(grad_input, grad_output, output_size, input_size);
}
static inline Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_output).upsample_bilinear2d_backward(grad_output, output_size, input_size);
}
static inline Tensor & upsample_trilinear3d_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_trilinear3d_out(output, self, output_size);
}
static inline Tensor upsample_trilinear3d(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_trilinear3d(self, output_size);
}
static inline Tensor & upsample_trilinear3d_forward_out(Tensor & output, const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_trilinear3d_forward_out(output, self, output_size);
}
static inline Tensor upsample_trilinear3d_forward(const Tensor & self, IntList output_size) {
    return infer_type(self).upsample_trilinear3d_forward(self, output_size);
}
static inline Tensor & upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_input).upsample_trilinear3d_backward_out(grad_input, grad_output, output_size, input_size);
}
static inline Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntList output_size, IntList input_size) {
    return infer_type(grad_output).upsample_trilinear3d_backward(grad_output, output_size, input_size);
}
static inline Tensor & upsample_nearest1d_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest1d(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d(self, scale_factor);
}
static inline Tensor & upsample_nearest1d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d_forward_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest1d_forward(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d_forward(self, scale_factor);
}
static inline Tensor & upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d_backward_out(grad_input, grad_output, self, scale_factor);
}
static inline Tensor upsample_nearest1d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest1d_backward(grad_output, self, scale_factor);
}
static inline Tensor & upsample_nearest2d_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest2d(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d(self, scale_factor);
}
static inline Tensor & upsample_nearest2d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d_forward_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest2d_forward(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d_forward(self, scale_factor);
}
static inline Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d_backward_out(grad_input, grad_output, self, scale_factor);
}
static inline Tensor upsample_nearest2d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest2d_backward(grad_output, self, scale_factor);
}
static inline Tensor & upsample_nearest3d_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest3d(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d(self, scale_factor);
}
static inline Tensor & upsample_nearest3d_forward_out(Tensor & output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d_forward_out(output, self, scale_factor);
}
static inline Tensor upsample_nearest3d_forward(const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d_forward(self, scale_factor);
}
static inline Tensor & upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d_backward_out(grad_input, grad_output, self, scale_factor);
}
static inline Tensor upsample_nearest3d_backward(const Tensor & grad_output, const Tensor & self, int64_t scale_factor) {
    return infer_type(self).upsample_nearest3d_backward(grad_output, self, scale_factor);
}
static inline Tensor & _sigmoid_out(Tensor & output, const Tensor & self) {
    return infer_type(self)._sigmoid_out(output, self);
}
static inline Tensor _sigmoid(const Tensor & self) {
    return infer_type(self)._sigmoid(self);
}
static inline Tensor & _sigmoid_forward_out(Tensor & output, const Tensor & self) {
    return infer_type(self)._sigmoid_forward_out(output, self);
}
static inline Tensor _sigmoid_forward(const Tensor & self) {
    return infer_type(self)._sigmoid_forward(self);
}
static inline Tensor & _sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_input)._sigmoid_backward_out(grad_input, grad_output, output);
}
static inline Tensor _sigmoid_backward(const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_output)._sigmoid_backward(grad_output, output);
}
static inline Tensor & _tanh_out(Tensor & output, const Tensor & self) {
    return infer_type(self)._tanh_out(output, self);
}
static inline Tensor _tanh(const Tensor & self) {
    return infer_type(self)._tanh(self);
}
static inline Tensor & _tanh_forward_out(Tensor & output, const Tensor & self) {
    return infer_type(self)._tanh_forward_out(output, self);
}
static inline Tensor _tanh_forward(const Tensor & self) {
    return infer_type(self)._tanh_forward(self);
}
static inline Tensor & _tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_input)._tanh_backward_out(grad_input, grad_output, output);
}
static inline Tensor _tanh_backward(const Tensor & grad_output, const Tensor & output) {
    return infer_type(grad_output)._tanh_backward(grad_output, output);
}
static inline Tensor & thnn_batch_norm_out(Tensor & output, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(self).thnn_batch_norm_out(output, self, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline Tensor thnn_batch_norm(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(self).thnn_batch_norm(self, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_batch_norm_forward_out(Tensor & output, Tensor & save_mean, Tensor & save_std, const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(self).thnn_batch_norm_forward_out(output, save_mean, save_std, self, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_batch_norm_forward(const Tensor & self, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps) {
    return infer_type(self).thnn_batch_norm_forward(self, weight, bias, running_mean, running_var, training, momentum, eps);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_batch_norm_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std) {
    return infer_type(self).thnn_batch_norm_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, running_mean, running_var, training, eps, save_mean, save_std);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_batch_norm_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, bool training, double eps, const Tensor & save_mean, const Tensor & save_std, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_batch_norm_backward(grad_output, self, weight, running_mean, running_var, training, eps, save_mean, save_std, output_mask);
}
static inline Tensor & thnn_conv_transpose2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose2d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline Tensor thnn_conv_transpose2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose2d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose2d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(self).thnn_conv_transpose2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv_transpose2d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, output_mask);
}
static inline Tensor & thnn_conv_transpose3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose3d_out(output, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline Tensor thnn_conv_transpose3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose3d(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, IntList dilation) {
    return infer_type(self).thnn_conv_transpose3d_forward(self, weight, kernel_size, bias, stride, padding, output_padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(self).thnn_conv_transpose3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList output_padding, IntList dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv_transpose3d_backward(grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, output_mask);
}
static inline Tensor & thnn_conv2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv2d_out(output, self, weight, kernel_size, bias, stride, padding);
}
static inline Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv2d(self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(self).thnn_conv2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv2d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
static inline Tensor & thnn_conv_depthwise2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_depthwise2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_depthwise2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor & thnn_conv_depthwise2d_forward_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_depthwise2d_forward_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_depthwise2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_depthwise2d_backward_out(grad_input, grad_weight, grad_output, self, weight, kernel_size, stride, padding, dilation);
}
static inline std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, std::array<bool,2> output_mask) {
    return infer_type(self).thnn_conv_depthwise2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, output_mask);
}
static inline Tensor & thnn_conv3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv3d_out(output, self, weight, kernel_size, bias, stride, padding);
}
static inline Tensor thnn_conv3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv3d(self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv3d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding) {
    return infer_type(self).thnn_conv3d_forward(self, weight, kernel_size, bias, stride, padding);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input) {
    return infer_type(self).thnn_conv3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv3d_backward(grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, output_mask);
}
static inline Tensor & thnn_conv_dilated2d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated2d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor thnn_conv_dilated2d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated2d(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated2d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated2d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(self).thnn_conv_dilated2d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv_dilated2d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
static inline Tensor & thnn_conv_dilated3d_out(Tensor & output, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated3d_out(output, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline Tensor thnn_conv_dilated3d(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated3d(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_forward_out(Tensor & output, Tensor & columns, Tensor & ones, const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated3d_forward_out(output, columns, ones, self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_forward(const Tensor & self, const Tensor & weight, IntList kernel_size, const Tensor & bias, IntList stride, IntList padding, IntList dilation) {
    return infer_type(self).thnn_conv_dilated3d_forward(self, weight, kernel_size, bias, stride, padding, dilation);
}
static inline std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv_dilated3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones) {
    return infer_type(self).thnn_conv_dilated3d_backward_out(grad_input, grad_weight, grad_bias, grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones);
}
static inline std::tuple<Tensor,Tensor,Tensor> thnn_conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntList kernel_size, IntList stride, IntList padding, IntList dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask) {
    return infer_type(self).thnn_conv_dilated3d_backward(grad_output, self, weight, kernel_size, stride, padding, dilation, columns, ones, output_mask);
}
static inline Tensor adaptive_avg_pool1d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_avg_pool1d(self, output_size);
}
static inline std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntList output_size) {
    return infer_type(self).adaptive_max_pool1d(self, output_size);
}
static inline bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol) {
    return infer_type(self).allclose(self, other, rtol, atol);
}
static inline Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self).addmv(self, mat, vec, beta, alpha);
}
static inline Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self).addmv_(self, mat, vec, beta, alpha);
}
static inline Tensor & addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha) {
    return infer_type(self).addmv_out(result, self, mat, vec, beta, alpha);
}
static inline Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self).addr(self, vec1, vec2, beta, alpha);
}
static inline Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self).addr_(self, vec1, vec2, beta, alpha);
}
static inline Tensor & addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha) {
    return infer_type(self).addr_out(result, self, vec1, vec2, beta, alpha);
}
static inline Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled) {
    return infer_type(input).batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled);
}
static inline Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator) {
    return infer_type(self).bernoulli_(self, p, generator);
}
static inline Tensor & bernoulli_(Tensor & self, double p, Generator * generator) {
    return infer_type(self).bernoulli_(self, p, generator);
}
static inline Tensor cat(TensorList tensors, int64_t dim) {
    return infer_type(tensors).cat(tensors, dim);
}
static inline Tensor & cat_out(Tensor & result, TensorList tensors, int64_t dim) {
    return infer_type(result).cat_out(result, tensors, dim);
}
static inline Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).sspaddmm(self, mat1, mat2, beta, alpha);
}
static inline Tensor & sspaddmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha) {
    return infer_type(self).sspaddmm_out(result, self, mat1, mat2, beta, alpha);
}
static inline std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim) {
    return infer_type(self).chunk(self, chunks, dim);
}
static inline bool cudnn_is_acceptable(const Tensor & self) {
    return infer_type(self).cudnn_is_acceptable(self);
}
static inline Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups) {
    return infer_type(input).convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups);
}
static inline Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled) {
    return infer_type(input)._convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
}
static inline Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding) {
    return infer_type(input)._convolution_nogroup(input, weight, bias, stride, padding, dilation, transposed, output_padding);
}
static inline std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntList stride, IntList padding, IntList dilation, bool transposed, IntList output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask) {
    return infer_type(self)._convolution_double_backward(ggI, ggW, ggb, gO, weight, self, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled, output_mask);
}
static inline Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {
    return infer_type(input).conv1d(input, weight, bias, stride, padding, dilation, groups);
}
static inline Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {
    return infer_type(input).conv2d(input, weight, bias, stride, padding, dilation, groups);
}
static inline Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList dilation, int64_t groups) {
    return infer_type(input).conv3d(input, weight, bias, stride, padding, dilation, groups);
}
static inline Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad) {
    return infer_type(self).conv_tbc(self, weight, bias, pad);
}
static inline std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad) {
    return infer_type(self).conv_tbc_backward(self, input, weight, bias, pad);
}
static inline Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {
    return infer_type(input).conv_transpose1d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
static inline Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {
    return infer_type(input).conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
static inline Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntList stride, IntList padding, IntList output_padding, int64_t groups, IntList dilation) {
    return infer_type(input).conv_transpose3d(input, weight, bias, stride, padding, output_padding, groups, dilation);
}
static inline Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
    return infer_type(theta).cudnn_affine_grid_generator(theta, N, C, H, W);
}
static inline Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W) {
    return infer_type(grad).cudnn_affine_grid_generator_backward(grad, N, C, H, W);
}
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon) {
    return infer_type(input).cudnn_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon);
}
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon) {
    return infer_type(input).cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var, epsilon);
}
static inline Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(self).cudnn_convolution(self, weight, bias, padding, stride, dilation, groups, benchmark, deterministic);
}
static inline Tensor cudnn_convolution_backward_input(IntList self_size, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(grad_output).cudnn_convolution_backward_input(self_size, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
    return infer_type(self).cudnn_convolution_backward(self, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
static inline Tensor cudnn_convolution_backward_bias(const Tensor & grad_output) {
    return infer_type(grad_output).cudnn_convolution_backward_bias(grad_output);
}
static inline Tensor cudnn_convolution_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(self).cudnn_convolution_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
static inline Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(self).cudnn_convolution_transpose(self, weight, bias, padding, output_padding, stride, dilation, groups, benchmark, deterministic);
}
static inline std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntList padding, IntList output_padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
    return infer_type(self).cudnn_convolution_transpose_backward(self, grad_output, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, output_mask);
}
static inline Tensor cudnn_convolution_transpose_backward_bias(const Tensor & grad_output) {
    return infer_type(grad_output).cudnn_convolution_transpose_backward_bias(grad_output);
}
static inline Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(grad_output).cudnn_convolution_transpose_backward_input(grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic);
}
static inline Tensor cudnn_convolution_transpose_backward_weight(IntList weight_size, const Tensor & grad_output, const Tensor & self, IntList padding, IntList stride, IntList dilation, int64_t groups, bool benchmark, bool deterministic) {
    return infer_type(self).cudnn_convolution_transpose_backward_weight(weight_size, grad_output, self, padding, stride, dilation, groups, benchmark, deterministic);
}
static inline Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid) {
    return infer_type(self).cudnn_grid_sampler(self, grid);
}
static inline std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output) {
    return infer_type(self).cudnn_grid_sampler_backward(self, grid, grad_output);
}
static inline Tensor det(const Tensor & self) {
    return infer_type(self).det(self);
}
static inline std::tuple<Tensor,Tensor,Tensor,Tensor> _det_with_svd(const Tensor & self) {
    return infer_type(self)._det_with_svd(self);
}
static inline Tensor dot(const Tensor & self, const Tensor & tensor) {
    return infer_type(self).dot(self, tensor);
}
static inline Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    return infer_type(weight).embedding(weight, indices, padding_idx, scale_grad_by_freq, sparse);
}
static inline Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
    return infer_type(grad).embedding_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq, sparse);
}
static inline Tensor embedding_dense_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    return infer_type(grad).embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
static inline Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type) {
    return infer_type(self).embedding_renorm_(self, indices, max_norm, norm_type);
}
static inline Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq) {
    return infer_type(grad).embedding_sparse_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
}
static inline Tensor empty_like(const Tensor & self) {
    return infer_type(self).empty_like(self);
}
static inline std::tuple<Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse) {
    return infer_type(weight).embedding_bag(weight, indices, offsets, scale_grad_by_freq, mode, sparse);
}
static inline Tensor embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse) {
    return infer_type(grad).embedding_bag_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode, sparse);
}
static inline Tensor embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) {
    return infer_type(grad).embedding_bag_sparse_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
static inline Tensor embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode) {
    return infer_type(grad).embedding_bag_dense_backward(grad, indices, offsets, offset2bag, bag_size, num_weights, scale_grad_by_freq, mode);
}
static inline Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, bool size_average, bool reduce) {
    return infer_type(self).hinge_embedding_loss(self, target, margin, size_average, reduce);
}
static inline Tensor ger(const Tensor & self, const Tensor & vec2) {
    return infer_type(self).ger(self, vec2);
}
static inline Tensor & ger_out(Tensor & result, const Tensor & self, const Tensor & vec2) {
    return infer_type(self).ger_out(result, self, vec2);
}
static inline Tensor index(const Tensor & self, TensorList indices) {
    return infer_type(self).index(self, indices);
}
static inline Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values) {
    return infer_type(self).index_put_(self, indices, values);
}
static inline bool is_cuda(const Tensor & self) {
    return infer_type(self).is_cuda(self);
}
static inline bool is_distributed(const Tensor & self) {
    return infer_type(self).is_distributed(self);
}
static inline bool is_floating_point(const Tensor & self) {
    return infer_type(self).is_floating_point(self);
}
static inline bool is_nonzero(const Tensor & self) {
    return infer_type(self).is_nonzero(self);
}
static inline bool is_same_size(const Tensor & self, const Tensor & other) {
    return infer_type(self).is_same_size(self, other);
}
static inline bool is_signed(const Tensor & self) {
    return infer_type(self).is_signed(self);
}
static inline bool is_sparse(const Tensor & self) {
    return infer_type(self).is_sparse(self);
}
static inline Tensor matmul(const Tensor & self, const Tensor & other) {
    return infer_type(self).matmul(self, other);
}
static inline std::tuple<Tensor,Tensor> max_pool1d(const Tensor & self, IntList kernel_size, IntList stride, IntList padding, IntList dilation, bool ceil_mode) {
    return infer_type(self).max_pool1d(self, kernel_size, stride, padding, dilation, ceil_mode);
}
static inline Tensor mm(const Tensor & self, const Tensor & mat2) {
    return infer_type(self).mm(self, mat2);
}
static inline Tensor & mm_out(Tensor & result, const Tensor & self, const Tensor & mat2) {
    return infer_type(self).mm_out(result, self, mat2);
}
static inline Tensor mv(const Tensor & self, const Tensor & vec) {
    return infer_type(self).mv(self, vec);
}
static inline Tensor & mv_out(Tensor & result, const Tensor & self, const Tensor & vec) {
    return infer_type(self).mv_out(result, self, vec);
}
static inline Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length) {
    return infer_type(self).narrow(self, dim, start, length);
}
static inline Tensor pin_memory(const Tensor & self) {
    return infer_type(self).pin_memory(self);
}
static inline Tensor rand_like(const Tensor & self) {
    return infer_type(self).rand_like(self);
}
static inline Tensor randn_like(const Tensor & self) {
    return infer_type(self).randn_like(self);
}
static inline Tensor repeat(const Tensor & self, IntList repeats) {
    return infer_type(self).repeat(self, repeats);
}
static inline std::tuple<Tensor,Tensor> RoiPooling2d_forward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale) {
    return infer_type(input).RoiPooling2d_forward(input, rois, pooledHeight, pooledWidth, spatialScale);
}
static inline Tensor RoiPooling2d_backward(const Tensor & input, const Tensor & rois, int64_t pooledHeight, int64_t pooledWidth, double spatialScale, const Tensor & gradOutput, const Tensor & argmaxes) {
    return infer_type(input).RoiPooling2d_backward(input, rois, pooledHeight, pooledWidth, spatialScale, gradOutput, argmaxes);
}
static inline Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu(self, lower, upper, training, generator);
}
static inline Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator) {
    return infer_type(self).rrelu_(self, lower, upper, training, generator);
}
static inline Tensor select(const Tensor & self, int64_t dim, int64_t index) {
    return infer_type(self).select(self, dim, index);
}
static inline Tensor selu(const Tensor & self) {
    return infer_type(self).selu(self);
}
static inline Tensor & selu_(Tensor & self) {
    return infer_type(self).selu_(self);
}
static inline int64_t size(const Tensor & self, int64_t dim) {
    return infer_type(self).size(self, dim);
}
static inline Tensor slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step) {
    return infer_type(self).slice(self, dim, start, end, step);
}
static inline std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim) {
    return infer_type(self).split(self, split_size, dim);
}
static inline Tensor squeeze(const Tensor & self) {
    return infer_type(self).squeeze(self);
}
static inline Tensor squeeze(const Tensor & self, int64_t dim) {
    return infer_type(self).squeeze(self, dim);
}
static inline Tensor & squeeze_(Tensor & self) {
    return infer_type(self).squeeze_(self);
}
static inline Tensor & squeeze_(Tensor & self, int64_t dim) {
    return infer_type(self).squeeze_(self, dim);
}
static inline Tensor stack(TensorList tensors, int64_t dim) {
    return infer_type(tensors).stack(tensors, dim);
}
static inline Tensor & stack_out(Tensor & result, TensorList tensors, int64_t dim) {
    return infer_type(result).stack_out(result, tensors, dim);
}
static inline Tensor stft(const Tensor & self, int64_t frame_length, int64_t hop, int64_t fft_size, bool return_onesided, const Tensor & window, int64_t pad_end) {
    return infer_type(self).stft(self, frame_length, hop, fft_size, return_onesided, window, pad_end);
}
static inline int64_t stride(const Tensor & self, int64_t dim) {
    return infer_type(self).stride(self, dim);
}
static inline Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1) {
    return infer_type(self).transpose_(self, dim0, dim1);
}
static inline Tensor & t_(Tensor & self) {
    return infer_type(self).t_(self);
}
static inline Tensor type_as(const Tensor & self, const Tensor & other) {
    return infer_type(self).type_as(self, other);
}
static inline Tensor unsqueeze(const Tensor & self, int64_t dim) {
    return infer_type(self).unsqueeze(self, dim);
}
static inline Tensor & unsqueeze_(Tensor & self, int64_t dim) {
    return infer_type(self).unsqueeze_(self, dim);
}
static inline Tensor view_as(const Tensor & self, const Tensor & other) {
    return infer_type(self).view_as(self, other);
}
static inline Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other) {
    return infer_type(self).where(condition, self, other);
}
static inline Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other) {
    return infer_type(self)._s_where(condition, self, other);
}
static inline Tensor _standard_gamma_grad(const Tensor & self, const Tensor & output) {
    return infer_type(self)._standard_gamma_grad(self, output);
}
static inline Tensor poisson(const Tensor & self, Generator * generator) {
    return infer_type(self).poisson(self, generator);
}
static inline Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional) {
    return infer_type(weight_arr)._cudnn_rnn_flatten_weight(weight_arr, weight_stride0, input_size, mode, hidden_size, num_layers, batch_first, bidirectional);
}
static inline std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state) {
    return infer_type(input)._cudnn_rnn(input, weight, weight_stride0, weight_buf, hx, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
}
static inline std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntList batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask) {
    return infer_type(input)._cudnn_rnn_backward(input, weight, weight_stride0, weight_buf, hx, cx, output, grad_output, grad_hy, grad_cy, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, reserve, output_mask);
}

}
