#include "THCStorage.hpp"
#include "THCGeneral.h"

#include "THCHalf.h"

#include <new>

#include "generic/THCStorage.cpp"
#include "THCGenerateAllTypes.h"

void THCStorage_resize(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator() != nullptr);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if (!self->resizable())
    THError("Trying to resize storage that is not resizable");

  size_t elementSize = self->elementSize();

  if(size == 0)
  {
    self->set_data_ptr(at::DataPtr(nullptr, at::Device(at::DeviceType::CUDA, device)));
    self->set_size(0);
  }
  else
  {
    at::DataPtr data =
      self->allocator()->allocate(size * elementSize);

    if (self->data_ptr()) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, THCStorage_getDevice(state, self));

      THCudaCheck(cudaMemcpyAsync(data.get(),
                                  self->data(),
                                  THMin(self->size(), size) * elementSize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_size(size);
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->getDevice();
}

THC_API THCStorage* THCStorage_new(
    THCState* state,
    at::ScalarType scalar_type) {
  THStorage* storage = new THStorage(
      scalar_type,
      0,
      state->cudaDeviceAllocator,
      true);
  return storage;
}
