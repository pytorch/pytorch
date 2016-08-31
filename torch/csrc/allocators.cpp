#include <Python.h>

#include "THP.h"

// Adapted from fblualib
void* ObjectPtrAllocator::malloc(long size) {
  return allocator->malloc(allocatorContext, size);
}


void* ObjectPtrAllocator::realloc(void* ptr, long size) {
  return allocator->realloc(allocatorContext, ptr, size);
}


void ObjectPtrAllocator::free(void* ptr) {
  object.release();
  allocator->free(allocatorContext, ptr);
  delete this;
}

void StorageWeakRefAllocator::free(void* ptr) {
  // All storage structs have the same structure and we just want to clear
  // the cdata field. Setting cdata to NULL will prevent the object from
  // calling free once more.
  THPFloatStorage *storage = (THPFloatStorage*)object.get();
  storage->cdata = nullptr;
  object = nullptr;
  allocator->free(allocatorContext, ptr);
  delete this;
}


#ifdef WITH_NUMPY
void* NumpyArrayAllocator::realloc(void* ptr, long size) {
  PyArrayObject *array_ptr = (PyArrayObject*)object.get();
  if (array_ptr && ptr == PyArray_DATA(array_ptr)) {
    void* newPtr = this->malloc(size);
    memcpy(newPtr, ptr, std::min(size, PyArray_NBYTES(array_ptr)));
    // Whee! We're done!
    object = nullptr;
    return newPtr;
  }
  return allocator->realloc(allocatorContext, ptr, size);
}


void NumpyArrayAllocator::free(void* ptr) {
  PyArrayObject *array_ptr = (PyArrayObject*)object.get();
  if (array_ptr && ptr == PyArray_DATA(array_ptr)) {
    object = nullptr;
    return;
  }
  allocator->free(allocatorContext, ptr);
  delete this;
}
#endif

template<typename T>
static void * malloc_wrapper(void *ctx, long size) {
  return ((T*)ctx)->malloc(size);
}

template<typename T>
static void * realloc_wrapper(void *ctx, void *ptr, long size) {
  return ((T*)ctx)->realloc(ptr, size);
}

template<typename T>
static void free_wrapper(void *ctx, void *ptr) {
  ((T*)ctx)->free(ptr);
}

THAllocator THObjectPtrAllocator = {
  malloc_wrapper<ObjectPtrAllocator>,
  realloc_wrapper<ObjectPtrAllocator>,
  free_wrapper<ObjectPtrAllocator>,
};

THAllocator THStorageWeakRefAllocator = {
  malloc_wrapper<StorageWeakRefAllocator>,
  realloc_wrapper<StorageWeakRefAllocator>,
  free_wrapper<StorageWeakRefAllocator>,
};

#ifdef WITH_NUMPY
THAllocator THNumpyArrayAllocator = {
  malloc_wrapper<NumpyArrayAllocator>,
  realloc_wrapper<NumpyArrayAllocator>,
  free_wrapper<NumpyArrayAllocator>,
};
#endif
