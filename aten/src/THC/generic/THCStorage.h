#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.h"
#else

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4

typedef struct THCStorage
{
    real *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;
} THCStorage;


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

/* takes ownership of data */
THC_API THCStorage* THCStorage_(newWithData)(THCState *state, real *data, ptrdiff_t size);

THC_API THCStorage* THCStorage_(newWithAllocator)(
  THCState *state, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);
THC_API THCStorage* THCStorage_(newWithDataAndAllocator)(
  THCState *state, real* data, ptrdiff_t size,
  THCDeviceAllocator* allocator,
  void *allocatorContext);

THC_API void THCStorage_(setFlag)(THCState *state, THCStorage *storage, const char flag);
THC_API void THCStorage_(clearFlag)(THCState *state, THCStorage *storage, const char flag);
THC_API void THCStorage_(retain)(THCState *state, THCStorage *storage);

THC_API void THCStorage_(free)(THCState *state, THCStorage *storage);
THC_API void THCStorage_(resize)(THCState *state, THCStorage *storage, ptrdiff_t size);
THC_API void THCStorage_(fill)(THCState *state, THCStorage *storage, real value);

THC_API int THCStorage_(getDevice)(THCState* state, const THCStorage* storage);

#endif
