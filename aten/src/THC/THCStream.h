#ifndef THC_STREAM_INC
#define THC_STREAM_INC

#include "THCGeneral.h"

/*
* Note: legacy API.
*
* Stream usage should be done through ATen/Context.h.
*/
typedef struct CUDAStreamInternals THCStream;

// Stream creation
THC_API THCStream* THCStream_defaultStream(int device);
THC_API THCStream* THCStream_new(int flags);
THC_API THCStream* THCStream_newWithPriority(int flags, int priority);

// Getters
THC_API cudaStream_t THCStream_stream(THCStream*);
THC_API int THCStream_device(THCStream*);

// Memory management
THC_API void THCStream_retain(THCStream*);
THC_API void THCStream_free(THCStream*);

#endif // THC_STREAM_INC
