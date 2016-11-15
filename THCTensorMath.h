#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THCTensor.h"
#include "THCGeneral.h"
#include "THCTensorCopy.h"

#include "generic/THCTensorMath.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathBlas.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMathMagma.h"
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

#include "generic/THCTensorMathScan.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorMasked.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorScatterGather.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorIndex.h"
#include "THCGenerateAllTypes.h"

#include "generic/THCTensorSort.h"
#include "THCGenerateAllTypes.h"

// MAGMA (i.e. CUDA implementation of LAPACK functions)
THC_API void THCudaTensor_geev(THCState *state, THCudaTensor *re_, THCudaTensor *rv_, THCudaTensor *a_, const char *jobvr);
THC_API void THCudaTensor_gesvd(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *a, const char *jobu);
THC_API void THCudaTensor_gesvd2(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *ra_, THCudaTensor *a, const char *jobu);
THC_API void THCudaTensor_getri(THCState *state, THCudaTensor *ra_, THCudaTensor *a);
THC_API void THCudaTensor_potri(THCState *state, THCudaTensor *ra_, THCudaTensor *a, const char *uplo);
THC_API void THCudaTensor_potrf(THCState *state, THCudaTensor *ra_, THCudaTensor *a, const char *uplo);
THC_API void THCudaTensor_potrs(THCState *state, THCudaTensor *rb_, THCudaTensor *a, THCudaTensor *b, const char *uplo);
THC_API void THCudaTensor_qr(THCState *state, THCudaTensor *rq_, THCudaTensor *rr_, THCudaTensor *a);


THC_API int THCudaByteTensor_logicalall(THCState *state, THCudaByteTensor *self);
THC_API int THCudaByteTensor_logicalany(THCState *state, THCudaByteTensor *self);

#endif
