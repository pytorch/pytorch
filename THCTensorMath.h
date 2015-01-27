#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"
#include "THCGeneral.h"

THC_API void THCudaTensor_fill(THCState *state, THCudaTensor *self, float value);
THC_API void THCudaTensor_zero(THCState *state, THCudaTensor *self);

THC_API void THCudaTensor_zeros(THCState *state, THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_ones(THCState *state, THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_reshape(THCState *state, THCudaTensor *r_, THCudaTensor *t, THLongStorage *size);
THC_API long THCudaTensor_numel(THCState *state, THCudaTensor *t);

THC_API void THCudaTensor_add(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_mul(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_div(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);


THC_API void THCudaTensor_cadd(THCState *state, THCudaTensor *self, THCudaTensor *src1, float value, THCudaTensor *src2);
THC_API void THCudaTensor_cmul(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cpow(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cdiv(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);

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

THC_API void THCudaTensor_addmv(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat, THCudaTensor *vec);
THC_API void THCudaTensor_addmm(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *mat1, THCudaTensor *mat2);
THC_API void THCudaTensor_addr(THCState *state, THCudaTensor *self, float beta, THCudaTensor *t, float alpha, THCudaTensor *vec1, THCudaTensor *vec2);
THC_API void THCudaTensor_baddbmm(THCState *state, THCudaTensor *result, float beta, THCudaTensor *t,
                                  float alpha, THCudaTensor *batch1, THCudaTensor *batch2);

THC_API void THCudaTensor_log(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_log1p(THCState *state, THCudaTensor *self, THCudaTensor *src);
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
THC_API void THCudaTensor_ceil(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_floor(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_abs(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_sign(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_round(THCState *state, THCudaTensor *self, THCudaTensor *src);
TH_API void THCudaTensor_atan2(THCState *state, THCudaTensor *r_, THCudaTensor *tx, THCudaTensor *ty);

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

THC_API void THCudaTensor_indexCopy(THCState *state, THCudaTensor *res_, int dim, THLongTensor *indices, THCudaTensor *src);
THC_API void THCudaTensor_indexFill(THCState *state, THCudaTensor *tensor, int dim, THLongTensor *index, float val);
THC_API void THCudaTensor_indexSelect(THCState *state, THCudaTensor *tensor, THCudaTensor *src, int dim, THLongTensor *index);


#endif
