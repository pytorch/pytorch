#include <Python.h>

#include "THP.h"

// Adapted from fblualib
void* NumpyArrayAllocator::malloc(long size) {
  return (*THDefaultAllocator.malloc)(nullptr, size);
}

void* NumpyArrayAllocator::realloc(void* ptr, long size) {
  PyArrayObject *array_ptr = (PyArrayObject*)array.get();
  if (array_ptr && ptr == PyArray_DATA(array_ptr)) {
    void* newPtr = this->malloc(size);
    memcpy(newPtr, ptr, std::min(size, PyArray_NBYTES(array_ptr)));
    // Whee! We're done!
    array = nullptr;
    return newPtr;
  }
  return (*THDefaultAllocator.realloc)(nullptr, ptr, size);
}

void NumpyArrayAllocator::free(void* ptr) {
  // We're relying on the slightly unsafe (and undocumented) behavior that
  // THStorage will only call the "free" method of the allocator once at the
  // end of its lifetime.
  PyArrayObject *array_ptr = (PyArrayObject*)array.get();
  if (array_ptr && ptr == PyArray_DATA(array_ptr)) {
    array = nullptr;
    return;
  }
  (*THDefaultAllocator.free)(nullptr, ptr);
  delete this;
}

static void * NumpyArrayAllocator_malloc(void *ctx, long size) {
  return ((NumpyArrayAllocator*)ctx)->malloc(size);
}

static void * NumpyArrayAllocator_realloc(void *ctx, void *ptr, long size) {
  return ((NumpyArrayAllocator*)ctx)->realloc(ptr, size);
}

static void NumpyArrayAllocator_free(void *ctx, void *ptr) {
  ((NumpyArrayAllocator*)ctx)->free(ptr);
}

THAllocator THNumpyArrayAllocator = {
  NumpyArrayAllocator_malloc,
  NumpyArrayAllocator_realloc,
  NumpyArrayAllocator_free
};
