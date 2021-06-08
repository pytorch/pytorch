#include <climits>
#include <c10/util/intrusive_ptr.h>

#include <TH/THStorageFunctions.hpp>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateAllTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateComplexTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateHalfType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateBoolType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateQTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateBFloat16Type.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateAllTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateComplexTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateHalfType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateBoolType.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateQTypes.h>

// NOLINTNEXTLINE(bugprone-suspicious-include)
#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateBFloat16Type.h>

THStorage* THStorage_new() {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
                           c10::StorageImpl::use_byte_size_t(),
                           0,
                           getTHDefaultAllocator(),
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
  if (storage->resizable()) {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr new_data;
    if (size_bytes != 0) {
      new_data = storage->allocator()->allocate(size_bytes);
    }
    at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
    ptrdiff_t old_capacity = storage->nbytes();
    storage->set_nbytes(size_bytes);
    if (old_data != nullptr) {
      ptrdiff_t copy_capacity = old_capacity;
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      if (storage->nbytes() < copy_capacity) {
        copy_capacity = storage->nbytes();
      }
      if (copy_capacity > 0) {
        memcpy(storage->data(), old_data.get(), copy_capacity);
      }
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}
