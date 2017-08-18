#include <Python.h>

#include "THP.h"

// Adapted from fblualib
void* ObjectPtrAllocator::malloc(ptrdiff_t size) {
  return allocator->malloc(allocatorContext, size);
}


void* ObjectPtrAllocator::realloc(void* ptr, ptrdiff_t size) {
  return allocator->realloc(allocatorContext, ptr, size);
}

void ObjectPtrAllocator::free(void* ptr) {
  object = nullptr;
  allocator->free(allocatorContext, ptr);
  delete this;
}

void StorageWeakRefAllocator::free(void* ptr) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject_SetAttrString(object.get(), "cdata", Py_None);
  object = nullptr;
  PyGILState_Release(gstate);
  allocator->free(allocatorContext, ptr);
  delete this;
}


#ifdef WITH_NUMPY
/**
 * Note [Numpy memory management]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * For efficiency reasons, when a user converts to/from numpy arrays,
 * we want to share the underlying storage.  This means that if we
 * turn a Numpy array into a Torch tensor, the Torch tensor must
 * keep the Numpy array alive, and vice versa for conversions in
 * the other direction.
 *
 * A Torch tensor keeps its backing Numpy array alive using the custom allocator
 * THNumpyArrayAllocator (backed by NumpyArrayAllocator), which holds a
 * THPObjectPointer to the Numpy PyArrayObject, and nulls it out upon free.
 * The relevant code is in torch/csrc/generic/Tensor.cpp.
 *
 * A Numpy array keeps its backing Torch tensor alive using the base object
 * <https://docs.scipy.org/doc/numpy-dev/reference/c-api.array.html#c.PyArray_SetBaseObject>
 * field of Numpy, which is Numpy's hook for allowing an external user to
 * manage memory.  The relevant code is in
 * torch/csrc/generic/methods/TensorSerialization.cwrap
 */

// See Note [Numpy memory management]
void* NumpyArrayAllocator::realloc(void* ptr, ptrdiff_t size) {
  PyArrayObject *array_ptr = (PyArrayObject*)object.get();
  if (array_ptr && ptr == PyArray_DATA(array_ptr)) {
    void* newPtr = this->malloc(size);
    memcpy(newPtr, ptr, std::min((size_t) size, (size_t) PyArray_NBYTES(array_ptr)));
    // Whee! We're done!
    object = nullptr;
    return newPtr;
  }
  return allocator->realloc(allocatorContext, ptr, size);
}

// See Note [Numpy memory management]
void NumpyArrayAllocator::free(void* ptr) {
  PyArrayObject *array_ptr = (PyArrayObject*)object.get();
  if (!array_ptr || ptr != PyArray_DATA(array_ptr))
    throw std::logic_error("invalid call to NumpyArrayAllocator::free()");
  object = nullptr;
  delete this;
}
#endif

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

#ifdef WITH_NUMPY
// See Note [Numpy memory management]
THAllocator THNumpyArrayAllocator = {
  malloc_wrapper<NumpyArrayAllocator>,
  realloc_wrapper<NumpyArrayAllocator>,
  free_wrapper<NumpyArrayAllocator>,
};
#endif

#ifdef WITH_CUDA
cudaError_t CudaStorageWeakRefAllocator::malloc(void** ptr, size_t size, cudaStream_t stream) {
  THError("CudaStorageWeakRefAllocator: malloc not supported");
  return cudaSuccess;
}

cudaError_t CudaStorageWeakRefAllocator::free(void* ptr) {
  PyGILState_STATE gstate = PyGILState_Ensure();
  PyObject_SetAttrString(object.get(), "cdata", Py_None);
  object = nullptr;
  PyGILState_Release(gstate);
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
