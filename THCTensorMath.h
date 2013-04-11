#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"

THC_API void THCudaTensor_fill(THCudaTensor *self, float value);
THC_API void THCudaTensor_zero(THCudaTensor *self);

THC_API void THCudaTensor_add(THCudaTensor *self, float value);
THC_API void THCudaTensor_mul(THCudaTensor *self, float value);
THC_API void THCudaTensor_div(THCudaTensor *self, float value);


THC_API void THCudaTensor_cadd(THCudaTensor *self, float value, THCudaTensor *src);  
THC_API void THCudaTensor_cadd_tst(THCudaTensor *self, THCudaTensor *src1, float value, THCudaTensor *src2);
THC_API void THCudaTensor_cmul(THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cdiv(THCudaTensor *self, THCudaTensor *src);

THC_API void THCudaTensor_addcmul(THCudaTensor *self, float value, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_addcdiv(THCudaTensor *self, float value, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_dot(THCudaTensor *self, THCudaTensor *src);
  
THC_API float THCudaTensor_minall(THCudaTensor *self);
THC_API float THCudaTensor_maxall(THCudaTensor *self);
THC_API float THCudaTensor_sumall(THCudaTensor *self);
THC_API void THCudaTensor_min(THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_max(THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_sum(THCudaTensor *self, THCudaTensor *src, long dim);

THC_API void THCudaTensor_addmv(THCudaTensor *self, float beta, float alpha, THCudaTensor *mat, THCudaTensor *vec);
THC_API void THCudaTensor_addmm(THCudaTensor *self, float beta, float alpha, THCudaTensor *mat1, THCudaTensor *mat2);
THC_API void THCudaTensor_addr(THCudaTensor *self, float alpha, THCudaTensor *vec1, THCudaTensor *vec2);

THC_API void THCudaTensor_log(THCudaTensor *self);
THC_API void THCudaTensor_log1p(THCudaTensor *self);
THC_API void THCudaTensor_exp(THCudaTensor *self);
THC_API void THCudaTensor_cos(THCudaTensor *self);
THC_API void THCudaTensor_acos(THCudaTensor *self);
THC_API void THCudaTensor_cosh(THCudaTensor *self);
THC_API void THCudaTensor_sin(THCudaTensor *self);
THC_API void THCudaTensor_asin(THCudaTensor *self);
THC_API void THCudaTensor_sinh(THCudaTensor *self);
THC_API void THCudaTensor_tan(THCudaTensor *self);
THC_API void THCudaTensor_atan(THCudaTensor *self);
THC_API void THCudaTensor_tanh(THCudaTensor *self);
THC_API void THCudaTensor_pow(THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_sqrt(THCudaTensor *self);
THC_API void THCudaTensor_ceil(THCudaTensor *self);
THC_API void THCudaTensor_floor(THCudaTensor *self);
THC_API void THCudaTensor_abs(THCudaTensor *self);
THC_API void THCudaTensor_sign(THCudaTensor *self, THCudaTensor *src);

THC_API void THCudaTensor_ltValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_gtValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_leValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_geValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_eqValue(THCudaTensor *self_, THCudaTensor *src, float value);
THC_API void THCudaTensor_neValue(THCudaTensor *self_, THCudaTensor *src, float value);

THC_API void THCudaTensor_ltTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_gtTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_leTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_geTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_eqTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_neTensor(THCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2);

THC_API float THCudaTensor_meanall(THCudaTensor *self);
THC_API void  THCudaTensor_mean(THCudaTensor *self, THCudaTensor *src, long dim);
THC_API float THCudaTensor_varall(THCudaTensor *self);
THC_API float THCudaTensor_stdall(THCudaTensor *self);
THC_API float THCudaTensor_normall(THCudaTensor *self, float value);
THC_API void  THCudaTensor_norm(THCudaTensor* self, THCudaTensor* src, float value, long dimension);
THC_API float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_rand(THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_randn(THCudaTensor *r_, THLongStorage *size);


#endif
