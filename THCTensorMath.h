#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"
#include "THCGeneral.h"

#include "generic/THCTensorMath.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathBlas.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathPairwise.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathPointwise.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathReduce.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathCompare.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathCompareT.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMasked.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorScatterGather.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorIndex.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorSort.h"
#include "THCGenerateAllTypes.h"

THC_API void THCudaTensor_tril(THCState *state, THCudaTensor *self, THCudaTensor *src, long k);
THC_API void THCudaTensor_triu(THCState *state, THCudaTensor *self, THCudaTensor *src, long k);
THC_API void THCudaTensor_diag(THCState *state, THCudaTensor *self, THCudaTensor *src, long k);
THC_API float THCudaTensor_trace(THCState *state, THCudaTensor *self);

THC_API void THCudaTensor_addcmul(THCState *state, THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_addcdiv(THCState *state, THCudaTensor *self, THCudaTensor* t, float value, THCudaTensor *src1, THCudaTensor *src2);

THC_API void THCudaTensor_cumsum(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);
THC_API void THCudaTensor_cumprod(THCState *state, THCudaTensor *self, THCudaTensor *src, long dim);

THC_API void THCudaTensor_cmin(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cmax(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);
THC_API void THCudaTensor_cminValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);
THC_API void THCudaTensor_cmaxValue(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_cross(THCState *state, THCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2, int dimension);
THC_API void THCudaTensor_clamp(THCState *state, THCudaTensor *self, THCudaTensor *src, float min_value, float max_value);

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

THC_API float THCudaTensor_dist(THCState *state, THCudaTensor *self, THCudaTensor *src, float value);

THC_API void THCudaTensor_rand(THCState *state, THCudaTensor *r_, THLongStorage *size);
THC_API void THCudaTensor_randn(THCState *state, THCudaTensor *r_, THLongStorage *size);

THC_API int THCudaByteTensor_logicalall(THCState *state, THCudaByteTensor *self);
THC_API int THCudaByteTensor_logicalany(THCState *state, THCudaByteTensor *self);

#endif
