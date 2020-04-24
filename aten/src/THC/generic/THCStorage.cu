#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCStorage.cu"
#else

void THCStorage_(fill)(THCState *state, THCStorage *self, scalar_t value)
{
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<scalar_t> self_data(THCStorage_(data)(state, self));
  thrust::fill(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
    thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
    self_data, self_data+self->numel(), value);
}

void THCStorage_(resize)(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THCStorage_resize(state, self, size);
}

int THCStorage_(getDevice)(THCState* state, const THCStorage* storage) {
  return THCStorage_getDevice(state, storage);
}

#endif
