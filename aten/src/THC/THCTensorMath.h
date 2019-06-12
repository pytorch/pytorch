#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include <THC/THCTensor.h>
#include <THC/THCGeneral.h>

#include <THC/generic/THCTensorMath.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathBlas.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathMagma.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathPairwise.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathPointwise.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathReduce.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathCompare.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathCompareT.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMathScan.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMasked.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorScatterGather.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorIndex.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorSort.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorMode.h>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCTensorTopK.h>
#include <THC/THCGenerateAllTypes.h>

THC_API int THCudaByteTensor_logicalAndAll(THCState *state, THCudaByteTensor *self);
THC_API int THCudaByteTensor_logicalAnyAll(THCState *state, THCudaByteTensor *self);

THC_API void THCudaByteTensor_logicalAnd(THCState *state, THCudaByteTensor *self, THCudaByteTensor *src, int dimension, int keepdim);
THC_API void THCudaByteTensor_logicalAny(THCState *state, THCudaByteTensor *self, THCudaByteTensor *src, int dimension, int keepdim);

#endif
