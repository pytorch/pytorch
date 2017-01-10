#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCStorage.cu"
#else

void THCStorage_(fill)(THCState *state, THCStorage *self, real value)
{
  THCThrustAllocator thrustAlloc(state);
  thrust::device_ptr<real> self_data(self->data);
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size, value);
}

void THCStorage_(resize)(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator != NULL);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  if (self->allocator->realloc) {
    THCHeapUpdate(state, (size - self->size) * sizeof(real));
    cudaError_t err = (*self->allocator->realloc)(
      self->allocatorContext,
      (void**)&(self->data),
      self->size * sizeof(real),
      size * sizeof(real), THCState_getCurrentStream(state));
    if (err != cudaSuccess) {
      THCHeapUpdate(state, (self->size - size) * sizeof(real));
      THCudaCheck(err);
    }
    self->size = size;
    self->device = device;
    return;
  }

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THCudaCheck(
        (*self->allocator->free)(self->allocatorContext, self->data));
      THCHeapUpdate(state, -self->size * sizeof(real));
    }
    self->data = NULL;
    self->size = 0;
    self->device = device;
  }
  else
  {
    real *data = NULL;
    // update heap *before* attempting malloc, to free space for the malloc
    THCHeapUpdate(state, size * sizeof(real));
    cudaError_t err =
      (*self->allocator->malloc)(self->allocatorContext,
                                 (void**)&(data),
                                 size * sizeof(real),
                                 THCState_getCurrentStream(state));
    if(err != cudaSuccess) {
      THCHeapUpdate(state, -size * sizeof(real));
    }
    THCudaCheck(err);

    if (self->data) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, self->device);

      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(real),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(self->flag & TH_STORAGE_FREEMEM) {
        THCudaCheck(
          (*self->allocator->free)(self->allocatorContext, self->data));
        THCHeapUpdate(state, -self->size * sizeof(real));
      }
    }

    self->data = data;
    self->size = size;
    self->device = device;
  }
}

THC_API int THCStorage_(getDevice)(THCState* state, const THCStorage* storage) {
  return storage->device;
}

#endif
