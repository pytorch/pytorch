#include "THCStream.h"

THC_API THCStream* THCStream_defaultStream(int device) {
  return at::CUDAStream_getDefaultStreamOnDevice(device);
}

THC_API THCStream* THCStream_new(int flags) { 
  return THCStream_newWithPriority(flags, 0);
}

THC_API THCStream* THCStream_newWithPriority(int flags, int priority) {
  return at::CUDAStream_createAndRetainWithOptions(flags, priority);
}

THC_API cudaStream_t THCStream_stream(THCStream* stream) {
  return at::CUDAStream_stream(stream);
}

THC_API int THCStream_device(THCStream* stream) { 
  return at::CUDAStream_device(stream);
}

THC_API void THCStream_retain(THCStream* stream) {
  at::CUDAStream_retain(stream); 
}

THC_API void THCStream_free(THCStream* stream) { 
  at::CUDAStream_free(stream); 
}


