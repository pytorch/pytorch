#include <climits>
#include <c10/util/intrusive_ptr.h>

#include <TH/THStorageFunctions.hpp>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateByteType.h>

#include <ATen/native/Resize.h>

THStorage* THStorage_new() {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           0,
                           c10::GetDefaultCPUAllocator(),
                           true)
                           .release();
  return storage;
}

// Free a non-weak pointer to THStorage
void THStorage_free(THStorage* storage) {
  if (!storage) {
    return;
  }
  c10::raw::intrusive_ptr::decref(storage);
}

void THStorage_retain(THStorage *storage)
{
  if (storage) {
    c10::raw::intrusive_ptr::incref(storage);
  }
}

void THStorage_resizeBytes(THStorage* storage, ptrdiff_t size_bytes) {
  at::native::resize_bytes_cpu(storage, size_bytes);
}
