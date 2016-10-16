#ifndef THC_STREAM_INC
#define THC_STREAM_INC

#include <cuda_runtime_api.h>
#include "THCGeneral.h"

typedef struct THCStream
{
    cudaStream_t stream;
    int device;
    int refcount;
} THCStream;


THC_API THCStream* THCStream_new(int flags);
THC_API void THCStream_free(THCStream* self);
THC_API void THCStream_retain(THCStream* self);

#endif // THC_STREAM_INC
