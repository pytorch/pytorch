#include "THCStorage.hpp"
#include "THCGeneral.h"

#include "THCHalf.h"

#include <new>

#include "generic/THCStorage.cpp"
#include "THCGenerateAllTypes.h"

THCStorage* THCStorage_new(THCState *state, at::ScalarType scalar_type)
{
  return THCStorage_newWithSize(state, scalar_type, 0);
}

THCStorage* THCStorage_newWithSize(THCState *state, at::ScalarType scalar_type, ptrdiff_t size)
{
  return THCStorage_newWithAllocator(
    state, scalar_type, size,
    state->cudaDeviceAllocator,
    state->cudaDeviceAllocator->state);
}

THCStorage* THCStorage_newWithAllocator(THCState *state,
                                        at::ScalarType scalar_type,
                                        ptrdiff_t size,
                                        THCDeviceAllocator* allocator,
                                        void* allocatorContext)
{
  THArgCheck(size >= 0, 2, "invalid size");
  int device;
  THCudaCheck(cudaGetDevice(&device));

  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  new (&storage->refcount) std::atomic<int>(1);
  storage->backend = at::kCUDA;
  storage->scalar_type = scalar_type;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  storage->size = size;
  storage->device = device;

  if(size > 0)
  {
    // update heap *before* attempting malloc, to free space for the malloc
    cudaError_t err =
      (*allocator->malloc)(allocatorContext,
                           (void**)&(storage->data_ptr),
                           size * at::elementSize(scalar_type),
                           THCState_getCurrentStreamOnDevice(state, device));
    if(err != cudaSuccess){
      free(storage);
    }
    THCudaCheck(err);
  } else {
    storage->data_ptr = NULL;
  }
  return storage;
}

void THCStorage_free(THCState *state, THCStorage *self)
{
  AT_ASSERT(self->backend == at::kCUDA);

  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--self->refcount == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      auto* thc_device_allocator = static_cast<THCDeviceAllocator*>(self->allocatorVoidPtr);
      THCudaCheck(
        (*thc_device_allocator->free)(self->allocatorContext, self->data_ptr));
    }
    if(self->flag & TH_STORAGE_VIEW) {
      THCStorage_free(state, self->view);
    }
    self->refcount.~atomic<int>();
    THFree(self);
  }
}

void THCStorage_resize(THCState *state, THCStorage *self, ptrdiff_t size)
{
  AT_ASSERT(self->backend == at::kCUDA);

  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocatorVoidPtr != NULL);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  size_t elementSize = at::elementSize(self->scalar_type);

  auto* thc_device_allocator = static_cast<THCDeviceAllocator*>(self->allocatorVoidPtr);

  if (thc_device_allocator->realloc) {
    void * data_ptr = self->data_ptr;
    cudaError_t err = (*thc_device_allocator->realloc)(
      self->allocatorContext,
      (void**)&(data_ptr),
      self->size * elementSize,
      size * elementSize, THCState_getCurrentStreamOnDevice(state, device));
    if (err != cudaSuccess) {
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
        (*thc_device_allocator->free)(self->allocatorContext, self->data_ptr));
    }
    self->data_ptr = NULL;
    self->size = 0;
    self->device = device;
  }
  else
  {
    void *data = NULL;
    cudaError_t err =
      (*thc_device_allocator->malloc)(self->allocatorContext,
                                 (void**)&(data),
                                 size * elementSize,
                                 THCState_getCurrentStreamOnDevice(state, device));
    THCudaCheck(err);

    if (self->data_ptr) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, self->device);

      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data_ptr,
                                  THMin(self->size, size) * elementSize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(self->flag & TH_STORAGE_FREEMEM) {
        THCudaCheck(
          (*thc_device_allocator->free)(self->allocatorContext, self->data_ptr));
      }
    }

    self->data_ptr = data;
    self->size = size;
    self->device = device;
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->device;
}

THCStorage* THCStorage_newWithData(THCState *state, at::ScalarType scalar_type, void *data, ptrdiff_t size)
{
  return THCStorage_newWithDataAndAllocator(state, scalar_type, data, size,
                                            state->cudaDeviceAllocator,
                                            state->cudaDeviceAllocator->state);
}

THCStorage* THCStorage_newWithDataAndAllocator(
  THCState *state, at::ScalarType scalar_type, void *data, ptrdiff_t size,
  THCDeviceAllocator *allocator, void *allocatorContext) {
  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  storage->backend = at::kCUDA;
  storage->scalar_type = scalar_type;
  storage->data_ptr = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  int device;
  if (data) {
    struct cudaPointerAttributes attr;
    THCudaCheck(cudaPointerGetAttributes(&attr, data));
    device = attr.device;
  } else {
    THCudaCheck(cudaGetDevice(&device));
  }
  storage->device = device;
  return storage;
}
