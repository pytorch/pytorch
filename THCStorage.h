#ifndef THC_STORAGE_INC
#define THC_STORAGE_INC

#include "THStorage.h"
#include "THCGeneral.h"

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4


typedef struct THCudaStorage
{
    float *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THCudaStorage *view;
} THCudaStorage;


THC_API float* THCudaStorage_data(THCState *state, const THCudaStorage*);
THC_API long THCudaStorage_size(THCState *state, const THCudaStorage*);

/* slow access -- checks everything */
THC_API void THCudaStorage_set(THCState *state, THCudaStorage*, long, float);
THC_API float THCudaStorage_get(THCState *state, const THCudaStorage*, long);

THC_API THCudaStorage* THCudaStorage_new(THCState *state);
THC_API THCudaStorage* THCudaStorage_newWithSize(THCState *state, long size);
THC_API THCudaStorage* THCudaStorage_newWithSize1(THCState *state, float);
THC_API THCudaStorage* THCudaStorage_newWithSize2(THCState *state, float, float);
THC_API THCudaStorage* THCudaStorage_newWithSize3(THCState *state, float, float, float);
THC_API THCudaStorage* THCudaStorage_newWithSize4(THCState *state, float, float, float, float);
THC_API THCudaStorage* THCudaStorage_newWithMapping(THCState *state, const char *filename, long size, int shared);

/* takes ownership of data */
THC_API THCudaStorage* THCudaStorage_newWithData(THCState *state, float *data, long size);

THC_API THCudaStorage* THCudaStorage_newWithAllocator(THCState *state, long size,
                                                      THAllocator* allocator,
                                                      void *allocatorContext);
THC_API THCudaStorage* THCudaStorage_newWithDataAndAllocator(
    THCState *state, float* data, long size, THAllocator* allocator, void *allocatorContext);

THC_API void THCudaStorage_setFlag(THCState *state, THCudaStorage *storage, const char flag);
THC_API void THCudaStorage_clearFlag(THCState *state, THCudaStorage *storage, const char flag);
THC_API void THCudaStorage_retain(THCState *state, THCudaStorage *storage);

THC_API void THCudaStorage_free(THCState *state, THCudaStorage *storage);
THC_API void THCudaStorage_resize(THCState *state, THCudaStorage *storage, long size);
THC_API void THCudaStorage_fill(THCState *state, THCudaStorage *storage, float value);

#endif
