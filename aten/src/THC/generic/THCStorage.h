#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.h"
#else

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

typedef struct THCStorage THCStorage;

namespace at {
  class CUDAStorageImpl;
  typedef CUDAStorageImpl CUDAByteStorageImpl;
  typedef CUDAStorageImpl CUDACharStorageImpl;
  typedef CUDAStorageImpl CUDAShortStorageImpl;
  typedef CUDAStorageImpl CUDAIntStorageImpl;
  typedef CUDAStorageImpl CUDALongStorageImpl;
  typedef CUDAStorageImpl CUDAFloatStorageImpl;
  typedef CUDAStorageImpl CUDADoubleStorageImpl;
  typedef CUDAStorageImpl CUDAHalfStorageImpl;
}

THC_API real* THCStorage_(data)(THCState *state, const at::CUDAStorageImpl*);
THC_API ptrdiff_t THCStorage_(size)(THCState *state, const at::CUDAStorageImpl*);
THC_API int THCStorage_(elementSize)(THCState *state);

/* slow access -- checks everything */
THC_API void THCStorage_(set)(THCState *state, at::CUDAStorageImpl*, ptrdiff_t, real);
THC_API real THCStorage_(get)(THCState *state, const at::CUDAStorageImpl*, ptrdiff_t);

THC_API at::CUDAStorageImpl* THCStorage_(new)(THCState *state);
THC_API at::CUDAStorageImpl* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size);
THC_API at::CUDAStorageImpl* THCStorage_(newWithSize1)(THCState *state, real);
THC_API at::CUDAStorageImpl* THCStorage_(newWithSize2)(THCState *state, real, real);
THC_API at::CUDAStorageImpl* THCStorage_(newWithSize3)(THCState *state, real, real, real);
THC_API at::CUDAStorageImpl* THCStorage_(newWithSize4)(THCState *state, real, real, real, real);
THC_API at::CUDAStorageImpl* THCStorage_(newWithMapping)(THCState *state, const char *filename, ptrdiff_t size, int shared);

/* takes ownership of data */
THC_API at::CUDAStorageImpl* THCStorage_(newWithData)(THCState *state, real *data, ptrdiff_t size);

THC_API at::CUDAStorageImpl* THCStorage_(newWithAllocator)(
  THCState *state, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);
THC_API at::CUDAStorageImpl* THCStorage_(newWithDataAndAllocator)(
  THCState *state, real* data, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);

THC_API void THCStorage_(setFlag)(THCState *state, at::CUDAStorageImpl *storage, const char flag);
THC_API void THCStorage_(clearFlag)(THCState *state, at::CUDAStorageImpl *storage, const char flag);
THC_API void THCStorage_(retain)(THCState *state, at::CUDAStorageImpl *storage);

/* used by StorageSharing */
THC_API int THCStorage_(retainIfLive)(THCState *state, at::CUDAStorageImpl *storage);

THC_API void THCStorage_(free)(THCState *state, at::CUDAStorageImpl *storage);
THC_API void THCStorage_(resize)(THCState *state, at::CUDAStorageImpl *storage, ptrdiff_t size);
THC_API void THCStorage_(fill)(THCState *state, at::CUDAStorageImpl *storage, real value);

THC_API int THCStorage_(getDevice)(THCState* state, const at::CUDAStorageImpl* storage);

#endif
