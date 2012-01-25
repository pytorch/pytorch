#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"

#define real float
#define Real Cuda

#define TH_GENERIC_FILE "generic/THStorage.h"
#include "generic/THStorage.h"
#undef TH_GENERIC_FILE

#define TH_GENERIC_FILE "generic/THStorageCopy.h"
#include "generic/THStorageCopy.h"
#undef TH_GENERIC_FILE

#undef real
#undef Real

void THByteStorage_copyCuda(THByteStorage *self, struct THCudaStorage *src);
void THCharStorage_copyCuda(THCharStorage *self, struct THCudaStorage *src);
void THShortStorage_copyCuda(THShortStorage *self, struct THCudaStorage *src);
void THIntStorage_copyCuda(THIntStorage *self, struct THCudaStorage *src);
void THLongStorage_copyCuda(THLongStorage *self, struct THCudaStorage *src);
void THFloatStorage_copyCuda(THFloatStorage *self, struct THCudaStorage *src);
void THDoubleStorage_copyCuda(THDoubleStorage *self, struct THCudaStorage *src);
void THCudaStorage_copyCuda(THCudaStorage *self, THCudaStorage *src);

#endif
