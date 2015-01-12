#ifndef TH_CUDA_TENSOR_COPY_INC
#define TH_CUDA_TENSOR_COPY_INC

#include "THCTensor.h"
#include "THCGeneral.h"

THC_API void THCudaTensor_copy(THCState *state, THCudaTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_copyByte(THCState *state, THCudaTensor *self, THByteTensor *src);
THC_API void THCudaTensor_copyChar(THCState *state, THCudaTensor *self, THCharTensor *src);
THC_API void THCudaTensor_copyShort(THCState *state, THCudaTensor *self, THShortTensor *src);
THC_API void THCudaTensor_copyInt(THCState *state, THCudaTensor *self, THIntTensor *src);
THC_API void THCudaTensor_copyLong(THCState *state, THCudaTensor *self, THLongTensor *src);
THC_API void THCudaTensor_copyFloat(THCState *state, THCudaTensor *self, THFloatTensor *src);
THC_API void THCudaTensor_copyDouble(THCState *state, THCudaTensor *self, THDoubleTensor *src);

THC_API void THByteTensor_copyCuda(THCState *state, THByteTensor *self, THCudaTensor *src);
THC_API void THCharTensor_copyCuda(THCState *state, THCharTensor *self, THCudaTensor *src);
THC_API void THShortTensor_copyCuda(THCState *state, THShortTensor *self, THCudaTensor *src);
THC_API void THIntTensor_copyCuda(THCState *state, THIntTensor *self, THCudaTensor *src);
THC_API void THLongTensor_copyCuda(THCState *state, THLongTensor *self, THCudaTensor *src);
THC_API void THFloatTensor_copyCuda(THCState *state, THFloatTensor *self, THCudaTensor *src);
THC_API void THDoubleTensor_copyCuda(THCState *state, THDoubleTensor *self, THCudaTensor *src);
THC_API void THCudaTensor_copyCuda(THCState *state, THCudaTensor *self, THCudaTensor *src);

#endif
