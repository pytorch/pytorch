#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"

#undef TH_API
#define TH_API THC_API
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
#undef TH_API
#ifdef WIN32
# define TH_API THC_EXTERNC __declspec(dllimport)
#else
# define TH_API THC_EXTERNC
#endif

THC_API void THByteStorage_copyCuda(THByteStorage *self, struct THCudaStorage *src);
THC_API void THCharStorage_copyCuda(THCharStorage *self, struct THCudaStorage *src);
THC_API void THShortStorage_copyCuda(THShortStorage *self, struct THCudaStorage *src);
THC_API void THIntStorage_copyCuda(THIntStorage *self, struct THCudaStorage *src);
THC_API void THLongStorage_copyCuda(THLongStorage *self, struct THCudaStorage *src);
THC_API void THFloatStorage_copyCuda(THFloatStorage *self, struct THCudaStorage *src);
THC_API void THDoubleStorage_copyCuda(THDoubleStorage *self, struct THCudaStorage *src);
THC_API void THCudaStorage_copyCuda(THCudaStorage *self, THCudaStorage *src);

#endif
