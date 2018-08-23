#include <climits>
#include <ATen/core/intrusive_ptr.h>

#include "THStorageFunctions.hpp"

#include "generic/THStorage.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.cpp"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateHalfType.h"

THStorage* THStorage_new(at::ScalarType scalar_type) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      scalar_type,
      0,
      getTHDefaultAllocator(),
      true).release();
  return storage;
}

// Free a non-weak pointer to THStorage
void THStorage_free(THStorage* storage) {
  if (!storage) {
    return;
  }
  storage->_raw_decref();
}

ptrdiff_t THStorage_size(const THStorage *self)
{
  return self->size();
}

void THStorage_retain(THStorage *storage)
{
  if (storage) {
    storage->_raw_incref();
  }
}

void THStorage_resize(THStorage* storage, ptrdiff_t size) {
  if (storage->resizable()) {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr new_data;
    if (size != 0) {
      new_data = storage->allocator()->allocate(storage->elementSize() * size);
    }
    at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
    ptrdiff_t old_size = storage->size();
    storage->set_size(size);
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if (storage->size() < copy_size) {
        copy_size = storage->size();
      }
      if (copy_size > 0) {
        memcpy(
            storage->data(),
            old_data.get(),
            storage->elementSize() * copy_size);
      }
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}
