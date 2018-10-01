#ifndef THC_STREAM_INC
#define THC_STREAM_INC

#include "THCGeneral.h"

/*
* Note: legacy API.
*
* Stream usage should be done through ATen/cuda/CUDAContext.h.
*/
typedef struct CUDAStreamInternals THCStream;

// Stream creation
THC_API THCStream* THCStream_defaultStream(int device);
THC_API THCStream* THCStream_new();

// Getters
THC_API cudaStream_t THCStream_stream(THCStream*);
THC_API int THCStream_device(THCStream*);

// Memory management 
// Note: these are no-ops, streams are no longer refcounted
THC_API void THCStream_retain(THCStream*);
THC_API void THCStream_free(THCStream*);

#endif // THC_STREAM_INC
