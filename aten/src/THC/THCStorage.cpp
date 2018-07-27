#include "THCStorage.hpp"
#include "THCGeneral.h"

#include "THCHalf.h"

#include <new>

#include "generic/THCStorage.cpp"
#include "THCGenerateAllTypes.h"

void THCStorage_resize(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator != nullptr);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    THError("Trying to resize storage that is not resizable");

  size_t elementSize = at::elementSize(self->scalar_type);

  if(size == 0)
  {
    self->data_ptr = at::DataPtr(nullptr, at::Device(at::kCUDA, device));
    self->size = 0;
  }
  else
  {
    at::DataPtr data =
      self->allocator->allocate(size * elementSize);

    if (self->data_ptr) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, THCStorage_getDevice(state, self));

      THCudaCheck(cudaMemcpyAsync(data.get(),
                                  self->data_ptr.get(),
                                  THMin(self->size, size) * elementSize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
    }

    // Destructively overwrite data_ptr
    self->data_ptr = std::move(data);
    self->size = size;
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->data_ptr.device().index();
}

THC_API THCStorage* THCStorage_new(
    THCState* state,
    at::ScalarType scalar_type) {
  THStorage* storage = new THStorage(
      scalar_type,
      0,
      state->cudaDeviceAllocator,
      TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE);
  return storage;
}
