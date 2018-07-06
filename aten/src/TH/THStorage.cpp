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

void THStorage_free(THStorage *storage) {
  AT_ASSERT(storage->backend == at::kCPU);

  if (!storage) {
    return;
  }

  if ((storage->flag & TH_STORAGE_REFCOUNTED) && (storage->refcount.load() > 0)) {
    if (--storage->refcount == 0) {
      if (storage->flag & TH_STORAGE_FREEMEM) {
        static_cast<THAllocator*>(storage->allocatorVoidPtr)->free(storage->allocatorContext, storage->data_ptr);
      }
      if (storage->flag & TH_STORAGE_VIEW) {
        THStorage_free(storage->view);
      }
      storage->refcount.~atomic<int>();
      THFree(storage);
    }
  }
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
  return THStorage_newWithAllocator(scalar_type, size, &THDefaultAllocator, nullptr);
}

THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                      THAllocator *allocator,
                                      void *allocatorContext)
{
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->backend = at::kCPU;
  storage->scalar_type = scalar_type;
  storage->data_ptr = allocator->malloc(allocatorContext, at::elementSize(scalar_type)*size);
  storage->size = size;
  new (&storage->refcount) std::atomic<int>(1);
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  storage->device = INT_MIN;  // device is not meaningful on CPU
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
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, flags);

  THStorage *storage = THStorage_newWithAllocator(scalar_type, size,
                                                  &THMapAllocator,
                                                  ctx);

  if (size <= 0) {
    storage->size = THMapAllocatorContext_size(ctx)/THStorage_elementSize(storage);
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

int THStorage_retainIfLive(THStorage *storage)
{
  // TODO: Check if TH_STORAGE_REFCOUNTED?
  int refcount = storage->refcount.load();
  while (refcount > 0) {
    if (storage->refcount.compare_exchange_strong(refcount, refcount + 1)) {
      return 1;
    }
    refcount = storage->refcount.load();
  }
  return 0;
}

THStorage* THStorage_newWithData(at::ScalarType scalar_type, void *data, ptrdiff_t size)
{
  return THStorage_newWithDataAndAllocator(scalar_type, data, size,
                                           &THDefaultAllocator, NULL);
}

THStorage* THStorage_newWithDataAndAllocator(at::ScalarType scalar_type,
                                             void* data, ptrdiff_t size,
                                             THAllocator* allocator,
                                             void* allocatorContext) {
  THStorage *storage = static_cast<THStorage*>(THAlloc(sizeof(THStorage)));
  storage->backend = at::kCPU;
  storage->scalar_type = scalar_type;
  storage->data_ptr = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  storage->allocatorVoidPtr = allocator;
  storage->allocatorContext = allocatorContext;
  storage->device = 0;
  return storage;
}

void THStorage_resize(THStorage *storage, ptrdiff_t size)
{
  AT_ASSERT(storage->backend == at::kCPU);

  auto* th_allocator = static_cast<THAllocator*>(storage->allocatorVoidPtr);

  if (storage->flag & TH_STORAGE_RESIZABLE)
  {
    if (th_allocator->realloc == nullptr) {
      /* case when the allocator does not have a realloc defined */
      void *old_data = storage->data_ptr;
      ptrdiff_t old_size = storage->size;
      if (size == 0) {
        storage->data_ptr = nullptr;
      } else {
        storage->data_ptr = th_allocator->malloc(
            storage->allocatorContext,
            at::elementSize(storage->scalar_type)*size);
      }
      storage->size = size;
      if (old_data != nullptr) {
        ptrdiff_t copy_size = old_size;
        if (storage->size < copy_size) {
          copy_size = storage->size;
        }
        if (copy_size > 0) {
          memcpy(storage->data_ptr, old_data, at::elementSize(storage->scalar_type)*copy_size);
        }
        th_allocator->free(storage->allocatorContext, old_data);
      }
    } else {
      storage->data_ptr = th_allocator->realloc(
              storage->allocatorContext,
              storage->data_ptr,
              at::elementSize(storage->scalar_type)*size);
      storage->size = size;
    }
  } else {
    THError("Trying to resize storage that is not resizable");
  }
}

void THStorage_swap(THStorage *storage1, THStorage *storage2)
{
#define SWAP(val) { std::swap(storage1->val, storage2->val); }
    SWAP(data_ptr);
    SWAP(size);
    SWAP(flag);
    // don't swap refcount!
    SWAP(allocatorVoidPtr);
    SWAP(allocatorContext);
    SWAP(view);
    SWAP(device);
#undef SWAP
}
