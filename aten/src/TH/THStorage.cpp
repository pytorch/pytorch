#include <climits>

#include "THStorage.hpp"

#include "generic/THStorage.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorage.cpp"
#include "THGenerateHalfType.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateAllTypes.h"

#include "generic/THStorageCopy.cpp"
#include "THGenerateHalfType.h"

// Free a non-weak pointer to THStorage
void THStorage_free(THStorage *storage) {
  if (!storage) {
    return;
  }

  if (storage->flag & TH_STORAGE_REFCOUNTED) {
    if (--storage->refcount == 0) {
      if (storage->finalizer) {
        (*storage->finalizer)();
      }
      storage->finalizer.~unique_ptr<THFinalizer>();
      storage->data_ptr.~DataPtr();
      THStorage_weakFree(storage);
    }
  }
}

// Manually retains a weak reference
void THStorage_weakRetain(THStorage *weak_storage) {
  weak_storage->weakcount++;
}

// Releases a weak reference
void THStorage_weakFree(THStorage *weak_storage) {
  if (--weak_storage->weakcount == 0) {
    weak_storage->refcount.~atomic<int>();
    weak_storage->weakcount.~atomic<int>();
    THFree(weak_storage);
  }
}

// Given a weak reference, returns a strong reference to a storage (which must
// be freed when done) or null if the storage is already dead.
THStorage* THStorage_weakLock(THStorage *weak_storage) {
  for (;;) {
    int refcount = weak_storage->refcount.load();
    if (refcount == 0) return nullptr;
    if (weak_storage->refcount.compare_exchange_strong(refcount, refcount + 1)) break;
  }
  return weak_storage;
}

THDescBuff THLongStorage_sizeDesc(const THLongStorage *size) {
  return _THSizeDesc(THLongStorage_data(size), size->size);
}

THLongStorage *THLongStorage_newInferSize(THLongStorage *size, ptrdiff_t nElement)
{
  ptrdiff_t total_size = (size->size > 0 ? 1 : 0);
  ptrdiff_t dim_infer = -1;
  ptrdiff_t i;
  for (i = 0; i < size->size; i++) {
    if (THLongStorage_data(size)[i] == -1) {
      THArgCheck(dim_infer == -1, 1, "only one dimension can be inferred");
      dim_infer = i;
    } else {
      total_size *= THLongStorage_data(size)[i];
    }
  }
  if (dim_infer != -1) {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(total_size > 0 && nElement % total_size == 0, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  } else {
    THDescBuff buf = THLongStorage_sizeDesc(size);
    THArgCheck(nElement == total_size, 2,
        "size '%s' is invalid for input with %td elements", buf.str, nElement);
  }
  THLongStorage* copy = THLongStorage_newWithSize(size->size);
  THLongStorage_copy(copy, size);
  if (dim_infer != -1) {
    THLongStorage_data(copy)[dim_infer] = nElement / total_size;
  }
  return copy;
}

THStorage* THStorage_new(at::ScalarType scalar_type)
{
  return THStorage_newWithSize(scalar_type, 0);
}

THStorage* THStorage_newWithSize(at::ScalarType scalar_type, ptrdiff_t size)
{
  return THStorage_newWithAllocator(scalar_type, size, getTHDefaultAllocator());
}

THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                      at::Allocator *allocator)
{
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->scalar_type = scalar_type;
  new (&storage->data_ptr) at::DataPtr(allocator->allocate(at::elementSize(scalar_type)*size));
  storage->size = size;
  new (&storage->refcount) std::atomic<int>(1);
  new (&storage->weakcount) std::atomic<int>(1); // from the strong reference
  new (&storage->finalizer) std::unique_ptr<THFinalizer>(nullptr);
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE;
  storage->allocator = allocator;
  return storage;
}

ptrdiff_t THStorage_size(const THStorage *self)
{
  return self->size;
}

size_t THStorage_elementSize(const THStorage *self)
{
  return at::elementSize(self->scalar_type);
}

THStorage* THStorage_newWithMapping(at::ScalarType scalar_type, const char *filename, ptrdiff_t size, int flags)
{
  size_t actual_size = -1;
  THStorage *storage = THStorage_newWithDataAndAllocator(scalar_type,
                                                         THMapAllocator::makeDataPtr(
                                                            filename,
                                                            flags,
                                                            size * at::elementSize(scalar_type),
                                                            &actual_size),
                                                         size,
                                                         /* allocator */ nullptr);

  if (size <= 0) {
    storage->size = actual_size/THStorage_elementSize(storage);
  }

  THStorage_clearFlag(storage, TH_STORAGE_RESIZABLE);

  return storage;
}

void THStorage_setFlag(THStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THStorage_clearFlag(THStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THStorage_retain(THStorage *storage)
{
  if (storage && (storage->flag & TH_STORAGE_REFCOUNTED)) {
    ++storage->refcount;
  }
}

/*
// I don't think you should ever call this
THStorage* THStorage_newWithData(at::ScalarType scalar_type, std::unique_ptr<at::BoundDeleter> data, ptrdiff_t size)
{
  return THStorage_newWithDataAndAllocator(scalar_type, data, size,
                                           getTHDefaultAllocator());
}
*/

THStorage* THStorage_newWithDataAndAllocator(at::ScalarType scalar_type,
                                             at::DataPtr&& data, ptrdiff_t size,
                                             THAllocator* allocator) {
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->scalar_type = scalar_type;
  new (&storage->data_ptr) at::DataPtr(std::move(data));
  storage->size = size;
  new (&storage->refcount) std::atomic<int>(1);
  new (&storage->weakcount) std::atomic<int>(1); // from the strong reference
  new (&storage->finalizer) std::unique_ptr<THFinalizer>(nullptr);
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE;
  storage->allocator = allocator;
  return storage;
}

void THStorage_resize(THStorage *storage, ptrdiff_t size)
{
  if (storage->flag & TH_STORAGE_RESIZABLE)
  {
    /* case when the allocator does not have a realloc defined */
    at::DataPtr old_data;
    std::swap(old_data, storage->data_ptr);
    ptrdiff_t old_size = storage->size;
    if (size != 0) {
      storage->data_ptr = storage->allocator->allocate(at::elementSize(storage->scalar_type)*size);
    }
    storage->size = size;
    if (old_data != nullptr) {
      ptrdiff_t copy_size = old_size;
      if (storage->size < copy_size) {
        copy_size = storage->size;
      }
      if (copy_size > 0) {
        memcpy(storage->data_ptr.get(), old_data.get(), at::elementSize(storage->scalar_type)*copy_size);
      }
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THStorage_swap(THStorage *storage1, THStorage *storage2)
{
#define SWAP(val) { std::swap(storage1->val, storage2->val); }
    SWAP(scalar_type);
    SWAP(data_ptr);
    SWAP(size);
    // don't swap refcount!
    SWAP(flag);
    SWAP(allocator);
    SWAP(finalizer);
#undef SWAP
}
