#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.h>

#include <TH/THHalf.h>

#include <new>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateAllTypes.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateComplexTypes.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateBoolType.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateBFloat16Type.h>

#include <c10/util/intrusive_ptr.h>

void THCStorage_resizeBytes(
    THCState* state,
    THCStorage* self,
    ptrdiff_t size_bytes) {
  THArgCheck(size_bytes >= 0, 2, "invalid size");
  THAssert(self->allocator() != nullptr);
  int device;
  THCudaCheck(cudaGetDevice(&device));

  if (!self->resizable())
    THError("Trying to resize storage that is not resizable");

  if (size_bytes == 0) {
    self->set_data_ptr_noswap(at::DataPtr(nullptr, at::Device(at::DeviceType::CUDA, device)));
    self->set_nbytes(0);
  } else {
    at::DataPtr data = self->allocator()->allocate(size_bytes);

    if (self->data_ptr()) {
      // Enable p2p access when the memcpy is across devices
      THCState_getPeerToPeerAccess(state, device, THCStorage_getDevice(state, self));

      THCudaCheck(cudaMemcpyAsync(
          data.get(),
          self->data(),
          THMin(self->nbytes(), size_bytes),
          cudaMemcpyDeviceToDevice,
          c10::cuda::getCurrentCUDAStream()));
    }

    // Destructively overwrite data_ptr
    self->set_data_ptr_noswap(std::move(data));
    self->set_nbytes(size_bytes);
  }
}

int THCStorage_getDevice(THCState* state, const THCStorage* storage) {
  return storage->device().index();
}

THCStorage* THCStorage_new(THCState* state) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           0,
                           c10::cuda::CUDACachingAllocator::get(),
                           true)
                           .release();
  return storage;
}
