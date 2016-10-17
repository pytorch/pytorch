#include "THCStream.h"

#include <cuda_runtime_api.h>
#include "THAtomic.h"


THCStream* THCStream_new(int flags)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THCudaCheck(cudaGetDevice(&self->device));
  THCudaCheck(cudaStreamCreateWithFlags(&self->stream, flags));
  return self;
}

void THCStream_free(THCStream* self)
{
  if (!self) {
    return;
  }
  if (THAtomicDecrementRef(&self->refcount)) {
    THCudaCheck(cudaStreamDestroy(self->stream));
    free(self);
  }
}

void THCStream_retain(THCStream* self)
{
  THAtomicIncrementRef(&self->refcount);
}
