#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"

TH_API void THCudaTensor_fill(THCudaTensor *self, float value);
TH_API void THCudaTensor_zero(THCudaTensor *self);

TH_API void THCudaTensor_add(THCudaTensor *self, float value);
TH_API void THCudaTensor_mul(THCudaTensor *self, float value);
TH_API void THCudaTensor_div(THCudaTensor *self, float value);

TH_API void THCudaTensor_cadd(THCudaTensor *self, float value, THCudaTensor *src);  
TH_API void THCudaTensor_cmul(THCudaTensor *self, THCudaTensor *src);
TH_API void THCudaTensor_cdiv(THCudaTensor *self, THCudaTensor *src);

TH_API void THCudaTensor_addcmul(THCudaTensor *self, float value, THCudaTensor *src1, THCudaTensor *src2);
TH_API void THCudaTensor_addcdiv(THCudaTensor *self, float value, THCudaTensor *src1, THCudaTensor *src2);

TH_API float THCudaTensor_dot(THCudaTensor *self, THCudaTensor *src);
  
TH_API float THCudaTensor_minall(THCudaTensor *self);
TH_API float THCudaTensor_maxall(THCudaTensor *self);
TH_API float THCudaTensor_sumall(THCudaTensor *self);

TH_API void THCudaTensor_addmv(THCudaTensor *self, float beta, float alpha, THCudaTensor *mat, THCudaTensor *vec);
TH_API void THCudaTensor_addmm(THCudaTensor *self, float beta, float alpha, THCudaTensor *mat1, THCudaTensor *mat2);
TH_API void THCudaTensor_addr(THCudaTensor *self, float alpha, THCudaTensor *vec1, THCudaTensor *vec2);

TH_API void THCudaTensor_log(THCudaTensor *self);
TH_API void THCudaTensor_log1p(THCudaTensor *self);
TH_API void THCudaTensor_exp(THCudaTensor *self);
TH_API void THCudaTensor_cos(THCudaTensor *self);
TH_API void THCudaTensor_acos(THCudaTensor *self);
TH_API void THCudaTensor_cosh(THCudaTensor *self);
TH_API void THCudaTensor_sin(THCudaTensor *self);
TH_API void THCudaTensor_asin(THCudaTensor *self);
TH_API void THCudaTensor_sinh(THCudaTensor *self);
TH_API void THCudaTensor_tan(THCudaTensor *self);
TH_API void THCudaTensor_atan(THCudaTensor *self);
TH_API void THCudaTensor_tanh(THCudaTensor *self);
TH_API void THCudaTensor_pow(THCudaTensor *self, float value);
TH_API void THCudaTensor_sqrt(THCudaTensor *self);
TH_API void THCudaTensor_ceil(THCudaTensor *self);
TH_API void THCudaTensor_floor(THCudaTensor *self);
TH_API void THCudaTensor_abs(THCudaTensor *self);

TH_API float THCudaTensor_meanall(THCudaTensor *self);
TH_API float THCudaTensor_varall(THCudaTensor *self);
TH_API float THCudaTensor_stdall(THCudaTensor *self);
TH_API float THCudaTensor_norm(THCudaTensor *self, float value);
TH_API float THCudaTensor_dist(THCudaTensor *self, THCudaTensor *src, float value);

TH_API void THCudaTensor_rand(THCudaTensor *r_, THLongStorage *size);
TH_API void THCudaTensor_randn(THCudaTensor *r_, THLongStorage *size);

#endif
