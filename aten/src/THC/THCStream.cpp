#include "THCStream.h"
#include "ATen/CUDAStream.h"

THC_API THCStream* THCStream_defaultStream(int device) {
  return at::detail::CUDAStream_getDefaultStreamOnDevice(device);
}

THC_API THCStream* THCStream_new(int flags) { 
  return THCStream_newWithPriority(flags, at::CUDAStream::DEFAULT_PRIORITY);
}

THC_API THCStream* THCStream_newWithPriority(int flags, int priority) {
  return at::detail::CUDAStream_createAndRetainWithOptions(flags, priority);
}

THC_API cudaStream_t THCStream_stream(THCStream* stream) {
  return at::detail::CUDAStream_stream(stream);
}

THC_API int THCStream_device(THCStream* stream) { 
  return at::detail::CUDAStream_device(stream);
}

THC_API void THCStream_retain(THCStream* stream) {
  at::detail::CUDAStream_retain(stream); 
}

THC_API void THCStream_free(THCStream* stream) { 
  at::detail::CUDAStream_free(stream); 
}


