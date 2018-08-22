#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.h"
#else

#define THCStorage THStorage

// These used to be distinct types; for some measure of backwards compatibility and documentation
// alias these to the single THCStorage type.
#define THCudaStorage       THCStorage
#define THCudaDoubleStorage THCStorage
#define THCudaHalfStorage   THCStorage
#define THCudaByteStorage   THCStorage
#define THCudaCharStorage   THCStorage
#define THCudaShortStorage  THCStorage
#define THCudaIntStorage    THCStorage
#define THCudaLongStorage   THCStorage

THC_API real* THCStorage_(data)(THCState *state, const THCStorage*);
THC_API ptrdiff_t THCStorage_(size)(THCState *state, const THCStorage*);
THC_API int THCStorage_(elementSize)(THCState *state);

/* slow access -- checks everything */
THC_API void THCStorage_(set)(THCState *state, THCStorage*, ptrdiff_t, real);
THC_API real THCStorage_(get)(THCState *state, const THCStorage*, ptrdiff_t);

THC_API THCStorage* THCStorage_(new)(THCState *state);
THC_API THCStorage* THCStorage_(newWithSize)(THCState *state, ptrdiff_t size);
THC_API THCStorage* THCStorage_(newWithSize1)(THCState *state, real);
THC_API THCStorage* THCStorage_(newWithSize2)(THCState *state, real, real);
THC_API THCStorage* THCStorage_(newWithSize3)(THCState *state, real, real, real);
THC_API THCStorage* THCStorage_(newWithSize4)(THCState *state, real, real, real, real);
THC_API THCStorage* THCStorage_(newWithMapping)(THCState *state, const char *filename, ptrdiff_t size, int shared);

#ifdef __cplusplus
THC_API THCStorage* THCStorage_(newWithAllocator)(
  THCState *state, ptrdiff_t size,
  at::Allocator* allocator);
THC_API THCStorage* THCStorage_(newWithDataAndAllocator)(
  THCState *state, at::DataPtr&& data, ptrdiff_t size,
  at::Allocator* allocator);
#endif

THC_API void THCStorage_(setFlag)(THCState *state, THCStorage *storage, const char flag);
THC_API void THCStorage_(clearFlag)(THCState *state, THCStorage *storage, const char flag);
THC_API void THCStorage_(retain)(THCState *state, THCStorage *storage);

THC_API void THCStorage_(free)(THCState *state, THCStorage *storage);
THC_API void THCStorage_(resize)(THCState *state, THCStorage *storage, ptrdiff_t size);
THC_API void THCStorage_(fill)(THCState *state, THCStorage *storage, real value);

THC_API int THCStorage_(getDevice)(THCState* state, const THCStorage* storage);

#endif
