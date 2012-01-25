#ifndef THC_TENSOR_INC
#define THC_TENSOR_INC

#include "THTensor.h"
#include "THCStorage.h"

#define real float
#define Real Cuda

#define TH_GENERIC_FILE "generic/THTensor.h"
#include "generic/THTensor.h"
#undef TH_GENERIC_FILE

#define TH_GENERIC_FILE "generic/THTensorCopy.h"
#include "generic/THTensorCopy.h"
#undef TH_GENERIC_FILE

#undef real
#undef Real

TH_API void THCudaTensor_fill(THCudaTensor *self, float value);

TH_API void THByteTensor_copyCuda(THByteTensor *self, THCudaTensor *src);
TH_API void THCharTensor_copyCuda(THCharTensor *self, THCudaTensor *src);
TH_API void THShortTensor_copyCuda(THShortTensor *self, THCudaTensor *src);
TH_API void THIntTensor_copyCuda(THIntTensor *self, THCudaTensor *src);
TH_API void THLongTensor_copyCuda(THLongTensor *self, THCudaTensor *src);
TH_API void THFloatTensor_copyCuda(THFloatTensor *self, THCudaTensor *src);
TH_API void THDoubleTensor_copyCuda(THDoubleTensor *self, THCudaTensor *src);
TH_API void THCudaTensor_copyCuda(THCudaTensor *self, THCudaTensor *src);

#endif
