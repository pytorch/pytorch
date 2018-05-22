#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cu"
#else

void THCStorage_(fill)(THCState *state, at::CUDAStorageImpl *self, real value)
{
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> self_data(self->data<real>());
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size(), value);
}

void THCStorage_(resize)(THCState *state, at::CUDAStorageImpl *self, ptrdiff_t size)
{
  self->resize(state, size, sizeof(real));
}

THC_API int THCStorage_(getDevice)(THCState* state, const at::CUDAStorageImpl* storage) {
  return storage->device();
}

#endif
