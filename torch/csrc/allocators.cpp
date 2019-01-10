#include "torch/csrc/python_headers.h"

#include "allocators.h"
#include "torch/csrc/utils/auto_gil.h"

// Adapted from fblualib
void* ObjectPtrAllocator::malloc(ptrdiff_t size) {
  return allocator->malloc(allocatorContext, size);
}


void* ObjectPtrAllocator::realloc(void* ptr, ptrdiff_t size) {
  return allocator->realloc(allocatorContext, ptr, size);
}

void ObjectPtrAllocator::free(void* ptr) {
  {
    AutoGIL gil;
    object = nullptr;
  }
  allocator->free(allocatorContext, ptr);
  delete this;
}

void StorageWeakRefAllocator::free(void* ptr) {
  {
    AutoGIL gil;
    PyObject_SetAttrString(object.get(), "cdata", Py_None);
    object = nullptr;
  }
  allocator->free(allocatorContext, ptr);
  delete this;
}

template<typename T>
static void * malloc_wrapper(void *ctx, ptrdiff_t size) {
  return ((T*)ctx)->malloc(size);
}

template<typename T>
static void * realloc_wrapper(void *ctx, void *ptr, ptrdiff_t size) {
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

#ifdef WITH_CUDA
cudaError_t CudaStorageWeakRefAllocator::malloc(void** ptr, size_t size, cudaStream_t stream) {
  THError("CudaStorageWeakRefAllocator: malloc not supported");
  return cudaSuccess;
}

cudaError_t CudaStorageWeakRefAllocator::free(void* ptr) {
  {
    AutoGIL gil;
    PyObject_SetAttrString(object.get(), "cdata", Py_None);
    object = nullptr;
  }
  cudaError_t err = allocator->free(allocatorContext, ptr);
  delete this;
  return err;
}

static cudaError_t cuda_malloc_wrapper(void *ctx, void** ptr, size_t size, cudaStream_t stream) {
  return ((CudaStorageWeakRefAllocator*)ctx)->malloc(ptr, size, stream);
}

static cudaError_t cuda_free_wrapper(void *ctx, void *ptr) {
  return ((CudaStorageWeakRefAllocator*)ctx)->free(ptr);
}

THCDeviceAllocator THCStorageWeakRefAllocator = {
  cuda_malloc_wrapper,
  NULL,
  cuda_free_wrapper,
  NULL,
};
#endif
