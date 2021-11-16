#include <THC/THCStorage.hpp>
#include <THC/THCGeneral.h>

#include <TH/THHalf.h>

#include <new>
#include <c10/cuda/CUDACachingAllocator.h>

#include <THC/generic/THCStorage.cpp>
#include <THC/THCGenerateByteType.h>

#include <c10/util/intrusive_ptr.h>
#include <ATen/native/cuda/Resize.h>

void THCStorage_resizeBytes(
    THCState* state,
    THCStorage* self,
    ptrdiff_t size_bytes_i) {
  TORCH_CHECK(!c10::overflows<size_t>(size_bytes_i),
              "Requested storage size (", size_bytes_i,
              ") cannot be represented as a size_t");
  const auto size_bytes = static_cast<size_t>(size_bytes_i);
  at::native::resize_bytes_cuda(self, size_bytes);
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
