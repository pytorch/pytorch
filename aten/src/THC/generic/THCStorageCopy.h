#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.h"
#else

/* Support for copy between different Storage types */

THC_API void THCStorage_(rawCopy)(THCState *state, THCStorage *storage, real *src);
THC_API void THCStorage_(copy)(THCState *state, THCStorage *storage, THCStorage *src);
THC_API void THCStorage_(copyByte)(THCState *state, THCStorage *storage, struct THByteStorage *src);
THC_API void THCStorage_(copyChar)(THCState *state, THCStorage *storage, struct THCharStorage *src);
THC_API void THCStorage_(copyShort)(THCState *state, THCStorage *storage, struct THShortStorage *src);
THC_API void THCStorage_(copyInt)(THCState *state, THCStorage *storage, struct THIntStorage *src);
THC_API void THCStorage_(copyLong)(THCState *state, THCStorage *storage, struct THLongStorage *src);
THC_API void THCStorage_(copyFloat)(THCState *state, THCStorage *storage, struct THFloatStorage *src);
THC_API void THCStorage_(copyDouble)(THCState *state, THCStorage *storage, struct THDoubleStorage *src);
THC_API void THCStorage_(copyHalf)(THCState *state, THCStorage *storage, struct THHalfStorage *src);

THC_API void THCStorage_(copyCudaByte)(THCState *state, THCStorage *storage, struct THCudaByteStorage *src);
THC_API void THCStorage_(copyCudaChar)(THCState *state, THCStorage *storage, struct THCudaCharStorage *src);
THC_API void THCStorage_(copyCudaShort)(THCState *state, THCStorage *storage, struct THCudaShortStorage *src);
THC_API void THCStorage_(copyCudaInt)(THCState *state, THCStorage *storage, struct THCudaIntStorage *src);
THC_API void THCStorage_(copyCudaLong)(THCState *state, THCStorage *storage, struct THCudaLongStorage *src);
THC_API void THCStorage_(copyCudaFloat)(THCState *state, THCStorage *storage, struct THCudaStorage *src);
THC_API void THCStorage_(copyCudaDouble)(THCState *state, THCStorage *storage, struct THCudaDoubleStorage *src);
#ifdef CUDA_HALF_TENSOR
THC_API void THCStorage_(copyCudaHalf)(THCState *state, THCStorage *storage, struct THCudaHalfStorage *src);
#endif

THC_API void TH_CONCAT_2(THByteStorage_copyCuda  , Real)(THCState *state, THByteStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THCharStorage_copyCuda  , Real)(THCState *state, THCharStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THShortStorage_copyCuda , Real)(THCState *state, THShortStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THIntStorage_copyCuda   , Real)(THCState *state, THIntStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THLongStorage_copyCuda  , Real)(THCState *state, THLongStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THFloatStorage_copyCuda , Real)(THCState *state, THFloatStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THDoubleStorage_copyCuda, Real)(THCState *state, THDoubleStorage *self, struct THCStorage *src);
THC_API void TH_CONCAT_2(THHalfStorage_copyCuda, Real)(THCState *state, THHalfStorage *self, struct THCStorage *src);

THC_API void THStorage_(copyCuda)(THCState *state, THStorage *self, THCStorage *src);
THC_API void THCStorage_(copyCuda)(THCState *state, THCStorage *self, THCStorage *src);
THC_API void THCStorage_(copyCPU)(THCState *state, THCStorage *self, THStorage *src);

#endif
