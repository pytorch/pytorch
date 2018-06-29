#include "torch/csrc/python_headers.h"

#include "allocators.h"
#include "torch/csrc/utils/auto_gil.h"

// Adapted from fblualib
void* ObjectPtrAllocator::malloc(ptrdiff_t size) {
  return allocator->allocate(allocatorContext, size);
}

void ObjectPtrAllocator::free(void* ptr) {
  {
    AutoGIL gil;
    object = nullptr;
  }
  allocator->deallocate(allocatorContext, ptr);
  delete this;
}

void StorageWeakRefAllocator::free(void* ptr) {
  {
    AutoGIL gil;
    PyObject_SetAttrString(object.get(), "cdata", Py_None);
    object = nullptr;
  }
  allocator->deallocate(allocatorContext, ptr);
  delete this;
}

template<typename T>
static void * malloc_wrapper(void *ctx, ptrdiff_t size) {
  return ((T*)ctx)->malloc(size);
}

template<typename T>
static void free_wrapper(void *ctx, void *ptr) {
  ((T*)ctx)->free(ptr);
}

struct THObjectPtrAllocator : at::Allocator {
  void* allocate(void* ctx, size_t size) const override {
    return malloc_wrapper<ObjectPtrAllocator>(ctx, size);
  }
  void deallocate(void* ctx, void* ptr) const override {
    return free_wrapper<ObjectPtrAllocator>(ctx, ptr);
  }
};

static THObjectPtrAllocator th_object_ptr_allocator;
at::Allocator* getTHObjectPtrAllocator() {
  return &th_object_ptr_allocator;
}

struct THStorageWeakRefAllocator : at::Allocator {
  void* allocate(void* ctx, size_t size) const override {
    return malloc_wrapper<StorageWeakRefAllocator>(ctx, size);
  }
  void deallocate(void* ctx, void* ptr) const override {
    return free_wrapper<StorageWeakRefAllocator>(ctx, ptr);
  }
};

static THStorageWeakRefAllocator th_storage_weak_ref_allocator;
at::Allocator* getTHStorageWeakRefAllocator() {
  return &th_storage_weak_ref_allocator;
}

#ifdef USE_CUDA
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
