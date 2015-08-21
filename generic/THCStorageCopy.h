#ifndef THC_STORAGE_COPY_INC
#define THC_STORAGE_COPY_INC

#include "THCStorage.h"
#include "THCGeneral.h"

/* Support for copy between different Storage types */

THC_API void THCudaStorage_rawCopy(THCState *state, THCudaStorage *storage, float *src);
THC_API void THCudaStorage_copy(THCState *state, THCudaStorage *storage, THCudaStorage *src);
THC_API void THCudaStorage_copyByte(THCState *state, THCudaStorage *storage, struct THByteStorage *src);
THC_API void THCudaStorage_copyChar(THCState *state, THCudaStorage *storage, struct THCharStorage *src);
THC_API void THCudaStorage_copyShort(THCState *state, THCudaStorage *storage, struct THShortStorage *src);
THC_API void THCudaStorage_copyInt(THCState *state, THCudaStorage *storage, struct THIntStorage *src);
THC_API void THCudaStorage_copyLong(THCState *state, THCudaStorage *storage, struct THLongStorage *src);
THC_API void THCudaStorage_copyFloat(THCState *state, THCudaStorage *storage, struct THFloatStorage *src);
THC_API void THCudaStorage_copyDouble(THCState *state, THCudaStorage *storage, struct THDoubleStorage *src);

THC_API void THByteStorage_copyCuda(THCState *state, THByteStorage *self, struct THCudaStorage *src);
THC_API void THCharStorage_copyCuda(THCState *state, THCharStorage *self, struct THCudaStorage *src);
THC_API void THShortStorage_copyCuda(THCState *state, THShortStorage *self, struct THCudaStorage *src);
THC_API void THIntStorage_copyCuda(THCState *state, THIntStorage *self, struct THCudaStorage *src);
THC_API void THLongStorage_copyCuda(THCState *state, THLongStorage *self, struct THCudaStorage *src);
THC_API void THFloatStorage_copyCuda(THCState *state, THFloatStorage *self, struct THCudaStorage *src);
THC_API void THDoubleStorage_copyCuda(THCState *state, THDoubleStorage *self, struct THCudaStorage *src);
THC_API void THCudaStorage_copyCuda(THCState *state, THCudaStorage *self, THCudaStorage *src);

#endif
