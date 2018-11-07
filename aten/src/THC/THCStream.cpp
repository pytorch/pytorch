#include "THCStream.h"
#include "ATen/cuda/CUDAStream.h"

THC_API THCStream* THCStream_defaultStream(int device) {
  return at::cuda::detail::CUDAStream_getDefaultStream(device);
}

THC_API THCStream* THCStream_new() {
  return at::cuda::detail::CUDAStream_getStreamFromPool();
}

THC_API cudaStream_t THCStream_stream(THCStream* stream) {
  return at::cuda::detail::CUDAStream_stream(stream);
}

THC_API int THCStream_device(THCStream* stream) {
  return at::cuda::detail::CUDAStream_device(stream);
}

THC_API void THCStream_retain(THCStream* stream) { }

THC_API void THCStream_free(THCStream* stream) { }


