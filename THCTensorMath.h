#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"
#include "THCGeneral.h"

#include "generic/THCTensorMath.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathPairwise.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathPointwise.h"
#include "THCGenerateAllTypes.h"


THC_API void THCudaTensor_tril(THCState *state, THCudaTensor *self, THCudaTensor *src, long k);
THC_API void THCudaTensor_triu(THCState *state, THCudaTensor *self, THCudaTensor *src, long k);

THC_API void THCudaTensor_addcmul(THCState *state, THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_addcdiv(THCState *state, THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_dot(THCState *state, THCudaTensor *self, THCudaTensor *src);

THC_API float THCudaTensor_minall(THCState *state, THCudaTensor *self);
THC_API float THCudaTensor_maxall(THCState *state, THCudaTensor *self);
THC_API float THCudaTensor_sumall(THCState *state, THCudaTensor *self);
THC_API float THCudaTensor_prodall(THCState *state, THCudaTensor *self);
THC_API void THCudaTensor_min(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dim);
THC_API void THCudaTensor_max(THCState *state, THCudaTensor *values, THCudaTensor *indices, THCudaTensor *src, long dim);
THC_API void THCudaTensor_sum(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_prod(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_cumsum(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_cumprod(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);

THC_API void THCudaTensor_cmin(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cmax(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cminValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_cmaxValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_addmv(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec);
THC_API void THCudaTensor_addmm(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat1, THCudaTensor *mat2);
THC_API void THCudaTensor_addr(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2);
THC_API void THCudaTensor_addbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
                                  float alpha, THCudaTensor *batch1, THCudaTensor *batch2);
THC_API void THCudaTensor_baddbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
                                  float alpha, THCudaTensor *batch1, THCudaTensor *batch2);

THC_API void THCudaTensor_log(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_log1p(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sigmoid(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_exp(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_cos(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_acos(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_cosh(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sin(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_asin(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sinh(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_tan(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_atan(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_tanh(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_pow(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_tpow(THCState *state, THCudaTensor *self, float value, THCudaTensor *src);
THC_API void THCudaTensor_clamp(THCState *state, THCudaTensor *self, THCudaTensor *src, float min_value, float max_value);
THC_API void THCudaTensor_sqrt(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_rsqrt(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_ceil(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_floor(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_abs(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_trunc(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_frac(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_neg(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_cinv(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sign(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_round(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_atan2(THCState *state, THCudaTensor *r_, THCudaTensor *tx, THCudaTensor *ty);
THC_API void THCudaTensor_lerp(THCState *state, THCudaTensor *result, THCudaTensor *a, THCudaTensor *b, float w);
THC_API void THCudaTensor_cross(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2, int dimension);

// MAGMA (i.e. CUDA implementation of LAPACK functions)
THC_API void THCudaTensor_gesv(THCState *state, THCudaTensor *rb_, THCudaTensor *ra_, THCudaTensor *b_, THCudaTensor *a_);
THC_API void THCudaTensor_gels(THCState *state, THCudaTensor *rb_, THCudaTensor *ra_, THCudaTensor *b_, THCudaTensor *a_);
THC_API void THCudaTensor_syev(THCState *state, THCudaTensor *re_, THCudaTensor *rv_, THCudaTensor *a_, const char *jobz, const char *uplo);
THC_API void THCudaTensor_geev(THCState *state, THCudaTensor *re_, THCudaTensor *rv_, THCudaTensor *a_, const char *jobvr);
THC_API void THCudaTensor_gesvd(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *a, const char *jobu);
THC_API void THCudaTensor_gesvd2(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *ra_, THCudaTensor *a, const char *jobu);
THC_API void THCudaTensor_getri(THCState *state, THCudaTensor *ra_, THCudaTensor *a);
THC_API void THCudaTensor_potri(THCState *state, THCudaTensor *ra_, THCudaTensor *a);
THC_API void THCudaTensor_potrf(THCState *state, THCudaTensor *ra_, THCudaTensor *a);
THC_API void THCudaTensor_potrs(THCState *state, THCudaTensor *rb_, THCudaTensor *a, THCudaTensor *b);
THC_API void THCudaTensor_qr(THCState *state, THCudaTensor *rq_, THCudaTensor *rr_, THCudaTensor *a);

THC_API void THCudaTensor_cat(THCState *state, THCudaTensor *result, THCudaTensor *ta, THCudaTensor *tb, int dimension);
THC_API void THCudaTensor_catArray(THCState *state, THCudaTensor *result, THCudaTensor **inputs, int numInputs, int dimension);

THC_API void THCudaTensor_ltValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_gtValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_leValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_geValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_eqValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_neValue(THCState *state, THCudaTensor *self_, THCudaTensor *src, float value);

THC_API void THCudaTensor_ltTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_gtTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_leTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_geTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_eqTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_neTensor(THCState *state, THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_meanall(THCState *state, THCudaTensor *self);
THC_API void  THCudaTensor_mean(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);
THC_API float THCudaTensor_varall(THCState *state, THCudaTensor *self);
THC_API void  THCudaTensor_var(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim, int flag);
THC_API float THCudaTensor_stdall(THCState *state, THCudaTensor *self);
THC_API void  THCudaTensor_std(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim, int flag);
THC_API float THCudaTensor_normall(THCState *state, THCudaTensor *self, float value);
THC_API void  THCudaTensor_norm(THCState *state, THCudaTensor* self, THCudaTensor* src, float value, long dimension);
THC_API void  THCudaTensor_renorm(THCState *state, THCudaTensor* self, THCudaTensor* src, float value, long dimension, float max_norm);
THC_API float THCudaTensor_dist(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size);

THC_API void THCudaTensor_indexCopy(THCState *state, THCudaTensor *res_, int dim, THCudaTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexAdd(THCState *state, THCudaTensor *res_, int dim, THCudaTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexFill(THCState *state, THCudaTensor *tensor, int dim, THCudaTensor *index, float val);
THC_API void THCudaTensor_indexSelect(THCState *state, THCudaTensor *tensor, THCudaTensor *src, int dim, THCudaTensor *index);

THC_API void THCudaTensor_indexCopy_long(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexAdd_long(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexFill_long(THCState *state, THCudaTensor *tensor, int dim, THLongTensor *index, float val);
THC_API void THCudaTensor_indexSelect_long(THCState *state, THCudaTensor *tensor, THCudaTensor *src, int dim, THLongTensor *index);

THC_API void THCudaTensor_maskedFill(THCState *state, THCudaTensor *tensor, THCudaTensor *mask, float value);
THC_API void THCudaTensor_maskedCopy(THCState *state, THCudaTensor *tensor, THCudaTensor *mask, THCudaTensor *src);
THC_API void THCudaTensor_maskedSelect(THCState *state, THCudaTensor *tensor, THCudaTensor *src, THCudaTensor *mask);

THC_API void THCudaTensor_maskedFillByte(THCState *state, THCudaTensor *tensor, THByteTensor *mask, float value);
THC_API void THCudaTensor_maskedCopyByte(THCState *state, THCudaTensor *tensor, THByteTensor *mask, THCudaTensor *src);
THC_API void THCudaTensor_maskedSelectByte(THCState *state, THCudaTensor *tensor, THCudaTensor *src, THByteTensor *mask);

THC_API void THCudaTensor_gather(THCState* state, THCudaTensor *tensor, THCudaTensor *src, int dim, THCudaTensor *index);
THC_API void THCudaTensor_scatter(THCState* state, THCudaTensor *tensor, int dim, THCudaTensor *index, THCudaTensor *src);
THC_API void THCudaTensor_scatterFill(THCState* state, THCudaTensor *tensor, int dim, THCudaTensor *index, float value);

THC_API int THCudaTensor_logicalall(THCState *state, THCudaTensor *self);
THC_API int THCudaTensor_logicalany(THCState *state, THCudaTensor *self);

#endif
