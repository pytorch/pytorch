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
    state->cudaDeviceAllocator);
}

THCStorage* THCStorage_newWithAllocator(THCState *state,
                                        at::ScalarType scalar_type,
                                        ptrdiff_t size,
                                        at::Allocator* allocator)
{
  THArgCheck(size >= 0, 2, "invalid size");
  int device;
  THCudaCheck(cudaGetDevice(&device));

  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  new (&storage->refcount) std::atomic<int>(1);
  new (&storage->weakcount) std::atomic<int>(1);
  new (&storage->finalizer) std::unique_ptr<THFinalizer>(nullptr);
  storage->backend = at::kCUDA;
  storage->scalar_type = scalar_type;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE;
  storage->allocator = allocator;
  storage->size = size;
  storage->device = device;

  at::SupervisedPtr ptr = nullptr;
  if (size > 0) {
    // update heap *before* attempting malloc, to free space for the malloc
    try {
      ptr = allocator->allocate(size * at::elementSize(scalar_type));
    } catch(...) {
      free(storage);
      throw;
    }
  }
  new (&storage->data_ptr) at::SupervisedPtr(std::move(ptr));
  return storage;
}

void THCStorage_free(THCState *state, THCStorage *storage)
{
  AT_ASSERT(storage->backend == at::kCUDA);

  if (storage->flag & TH_STORAGE_REFCOUNTED) {
    if (--storage->refcount == 0) {
      if (storage->finalizer) {
        (*storage->finalizer)();
      }
      storage->finalizer.~unique_ptr<THFinalizer>();
      storage->data_ptr.~unique_ptr<void, at::BoundDeleter>();
      if (storage->flag & TH_STORAGE_VIEW) {
        THCStorage_free(state, storage->view);
      }
      THStorage_weakFree(storage);
    }
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
    self->data_ptr = nullptr;
    self->size = 0;
    self->device = device;
  }
  else
  {
    at::SupervisedPtr data =
      self->allocator->allocate(size * elementSize);

    if (self->data_ptr) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, self->device);

      THCudaCheck(cudaMemcpyAsync(data.get(),
                                  self->data_ptr.get(),
                                  THMin(self->size, size) * elementSize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
    }

    // Destructively overwrite data_ptr
    self->data_ptr = std::move(data);
    self->size = size;
    self->device = device;
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->device;
}

THCStorage* THCStorage_newWithDataAndAllocator(
  THCState *state, at::ScalarType scalar_type, at::SupervisedPtr&& data, ptrdiff_t size,
  at::Allocator *allocator) {
  THCStorage *storage = (THCStorage*)THAlloc(sizeof(THCStorage));
  memset(storage, 0, sizeof(THCStorage));
  storage->backend = at::kCUDA;
  storage->scalar_type = scalar_type;
  new (&storage->data_ptr) at::SupervisedPtr(std::move(data));
  storage->size = size;
  new (&storage->refcount) std::atomic<int>(1);
  new (&storage->weakcount) std::atomic<int>(1);
  new (&storage->finalizer) std::unique_ptr<THFinalizer>(nullptr);
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE;
  storage->allocator = allocator;
  int device;
  if (storage->data_ptr) {
    struct cudaPointerAttributes attr;
    THCudaCheck(cudaPointerGetAttributes(&attr, storage->data_ptr.get()));
    device = attr.device;
  } else {
    THCudaCheck(cudaGetDevice(&device));
  }
  storage->device = device;
  return storage;
}
