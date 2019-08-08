#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.h>

#include <TH/THHalf.h>

#include <new>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateBFloat16Type.h>

#include <c10/util/intrusive_ptr.h>

void THCStorage_resize(THCState *state, THCStorage *self, ptrdiff_t size)
{
  THArgCheck(size >= 0, 2, "invalid size");
  THAssert(self->allocator() != nullptr);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if (!self->resizable())
    THError("Trying to resize storage that is not resizable");

  size_t itemsize = self->itemsize();

  if(size == 0)
  {
    self->set_data_ptr(at::DataPtr(nullptr, at::Device(at::DeviceType::CUDA, device)));
    self->set_numel(0);
  }
  else
  {
    at::DataPtr data =
      self->allocator()->allocate(size * itemsize);

    if (self->data_ptr()) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, THCStorage_getDevice(state, self));

      THCudaCheck(cudaMemcpyAsync(data.get(),
                                  self->data(),
                                  THMin(self->numel(), size) * itemsize,
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr(std::move(data));
    self->set_numel(size);
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->device().index();
}

THC_API THCStorage* THCStorage_new(
    THCState* state,
    caffe2::TypeMeta data_type) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      data_type,
      0,
      state->cudaDeviceAllocator,
      true).release();
  return storage;
}
