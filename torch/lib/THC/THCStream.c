#include "THCStream.h"

#include <cuda_runtime_api.h>
#include "THAtomic.h"

#define MAX_DEVICES 256
static THCStream default_streams[MAX_DEVICES];


THCStream* THCStream_new(int flags)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THCudaCheck(cudaGetDevice(&self->device));
  THCudaCheck(cudaStreamCreateWithFlags(&self->stream, flags));
  return self;
}

THC_API THCStream* THCStream_defaultStream(int device)
{
  // default streams aren't refcounted
  THAssert(device >= 0 && device < MAX_DEVICES);
  THCStream* stream = &default_streams[device];
  stream->device = device;
  return stream;
}

THCStream* THCStream_newWithPriority(int flags, int priority)
{
  THCStream* self = (THCStream*) malloc(sizeof(THCStream));
  self->refcount = 1;
  THCudaCheck(cudaGetDevice(&self->device));
  THCudaCheck(cudaStreamCreateWithPriority(&self->stream, flags, priority));
  return self;
}

void THCStream_free(THCStream* self)
{
  if (!self || !self->stream) {
    return;
  }
  if (THAtomicDecrementRef(&self->refcount)) {
    THCudaCheckWarn(cudaStreamDestroy(self->stream));
    free(self);
  }
}

void THCStream_retain(THCStream* self)
{
  if (self->stream) {
    THAtomicIncrementRef(&self->refcount);
  }
}
