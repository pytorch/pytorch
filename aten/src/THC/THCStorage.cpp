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
    state->cudaDeviceAllocatorState);
}

THCStorage* THCStorage_newWithAllocator(THCState *state,
                                        at::ScalarType scalar_type,
                                        ptrdiff_t size,
                                        at::Allocator* allocator,
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
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  storage->size = size;
  storage->device = device;

  if(size > 0)
  {
    // update heap *before* attempting malloc, to free space for the malloc
    try {
      storage->data_ptr = allocator->allocate(allocatorContext,
                           size * at::elementSize(scalar_type));
    } catch(...) {
      free(storage);
      throw;
    }
  } else {
    storage->data_ptr = NULL;
  }
  return storage;
}

void THCStorage_retain(THCState *state, THCStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    self->refcount++;
}

void THCStorage_free(THCState *state, THCStorage *self)
{
  AT_ASSERT(self->backend == at::kCUDA);

  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (--self->refcount == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      self->allocator->deallocate(self->allocatorContext, self->data_ptr);
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
  THAssert(self->allocator != nullptr);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  size_t elementSize = at::elementSize(self->scalar_type);

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      self->allocator->deallocate(self->allocatorContext, self->data_ptr);
    }
    self->data_ptr = NULL;
    self->size = 0;
    self->device = device;
  }
  else
  {
    void *data =
      self->allocator->allocate(self->allocatorContext, size * elementSize);

    if (self->data_ptr) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, self->device);

      THCudaCheck(cudaMemcpyAsync(data,
                                  self->data_ptr,
                                  THMin(self->size, size) * elementSize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      if(self->flag & TH_STORAGE_FREEMEM) {
        self->allocator->deallocate(self->allocatorContext, self->data_ptr);
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
