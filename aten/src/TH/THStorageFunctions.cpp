#include <climits>
#include <c10/util/intrusive_ptr.h>

#include <TH/THStorageFunctions.hpp>

#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateQTypes.h>

#include <TH/generic/THStorage.cpp>
#include <TH/THGenerateBFloat16Type.h>

#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateAllTypes.h>

#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateHalfType.h>

#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateBoolType.h>

#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateQTypes.h>

#include <TH/generic/THStorageCopy.cpp>
#include <TH/THGenerateBFloat16Type.h>

THStorage* THStorage_new(caffe2::TypeMeta data_type) {
  THStorage* storage = c10::make_intrusive<at::StorageImpl>(
      data_type,
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
  c10::raw::intrusive_ptr::decref(storage);
}

ptrdiff_t THStorage_size(const THStorage *self)
{
  return self->numel();
}

void THStorage_retain(THStorage *storage)
{
  if (storage) {
    c10::raw::intrusive_ptr::incref(storage);
  }
}

void THStorage_resize(THStorage* storage, ptrdiff_t size) {
  if (storage->resizable()) {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr new_data;
    if (size != 0) {
      new_data = storage->allocator()->allocate(storage->itemsize() * size);
    }
    at::DataPtr old_data = storage->set_data_ptr(std::move(new_data));
    ptrdiff_t old_size = storage->numel();
    storage->set_numel(size);
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if (storage->numel() < copy_size) {
        copy_size = storage->numel();
      }
      if (copy_size > 0) {
        memcpy(
            storage->data(),
            old_data.get(),
            storage->itemsize() * copy_size);
      }
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}
