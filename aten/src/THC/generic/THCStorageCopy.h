#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorageCopy.h"
#else

/* Support for copy between different Storage types */

THC_API void THCStorage_(rawCopy)(THCState *state, at::CUDAStorageImpl *storage, real *src);
THC_API void THCStorage_(copy)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAStorageImpl *src);
THC_API void THCStorage_(copyByte)(THCState *state, at::CUDAStorageImpl *storage, at::ByteStorageImpl *src);
THC_API void THCStorage_(copyChar)(THCState *state, at::CUDAStorageImpl *storage, at::CharStorageImpl *src);
THC_API void THCStorage_(copyShort)(THCState *state, at::CUDAStorageImpl *storage, at::ShortStorageImpl *src);
THC_API void THCStorage_(copyInt)(THCState *state, at::CUDAStorageImpl *storage, at::IntStorageImpl *src);
THC_API void THCStorage_(copyLong)(THCState *state, at::CUDAStorageImpl *storage, at::LongStorageImpl *src);
THC_API void THCStorage_(copyFloat)(THCState *state, at::CUDAStorageImpl *storage, at::FloatStorageImpl *src);
THC_API void THCStorage_(copyDouble)(THCState *state, at::CUDAStorageImpl *storage, at::DoubleStorageImpl *src);
THC_API void THCStorage_(copyHalf)(THCState *state, at::CUDAStorageImpl *storage, at::HalfStorageImpl *src);

THC_API void THCStorage_(copyCudaByte)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAByteStorageImpl *src);
THC_API void THCStorage_(copyCudaChar)(THCState *state, at::CUDAStorageImpl *storage, at::CUDACharStorageImpl *src);
THC_API void THCStorage_(copyCudaShort)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAShortStorageImpl *src);
THC_API void THCStorage_(copyCudaInt)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAIntStorageImpl *src);
THC_API void THCStorage_(copyCudaLong)(THCState *state, at::CUDAStorageImpl *storage, at::CUDALongStorageImpl *src);
THC_API void THCStorage_(copyCudaFloat)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAFloatStorageImpl *src);
THC_API void THCStorage_(copyCudaDouble)(THCState *state, at::CUDAStorageImpl *storage, at::CUDADoubleStorageImpl *src);
#ifdef CUDA_HALF_TENSOR
THC_API void THCStorage_(copyCudaHalf)(THCState *state, at::CUDAStorageImpl *storage, at::CUDAHalfStorageImpl *src);
#endif

THC_API void TH_CONCAT_2(THByteStorage_copyCuda  , Real)(THCState *state, at::ByteStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THCharStorage_copyCuda  , Real)(THCState *state, at::CharStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THShortStorage_copyCuda , Real)(THCState *state, at::ShortStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THIntStorage_copyCuda   , Real)(THCState *state, at::IntStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THLongStorage_copyCuda  , Real)(THCState *state, at::LongStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THFloatStorage_copyCuda , Real)(THCState *state, at::FloatStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THDoubleStorage_copyCuda, Real)(THCState *state, at::DoubleStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void TH_CONCAT_2(THHalfStorage_copyCuda, Real)(THCState *state, at::HalfStorageImpl *self, at::CUDAStorageImpl *src);

THC_API void THStorage_(copyCuda)(THCState *state, at::StorageImpl *self, at::CUDAStorageImpl *src);
THC_API void THCStorage_(copyCuda)(THCState *state, at::CUDAStorageImpl *self, at::CUDAStorageImpl *src);
THC_API void THCStorage_(copyCPU)(THCState *state, at::CUDAStorageImpl *self, at::StorageImpl *src);

#endif
