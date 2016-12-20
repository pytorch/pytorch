#ifndef THP_NUMPY_INC
#define THP_NUMPY_INC

#include <type_traits>
#include <memory>

#ifdef WITH_CUDA
#include <THC/THC.h>
#endif

#ifdef WITH_NUMPY

#ifndef WITH_NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#endif

// Adapted from fblualib
class ObjectPtrAllocator {
public:
  ObjectPtrAllocator(PyObject *wrapped_object):
      ObjectPtrAllocator(wrapped_object, &THDefaultAllocator, nullptr) {}

  ObjectPtrAllocator(PyObject *wrapped_object, THAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

  void* malloc(long size);
  void* realloc(void* ptr, long size);
  void free(void* ptr);

  THPObjectPtr object;
  THAllocator *allocator;
  void *allocatorContext;
};

class StorageWeakRefAllocator: public ObjectPtrAllocator {
public:
  StorageWeakRefAllocator(PyObject *wrapped_object, THAllocator *alloc, void *ctx):
    ObjectPtrAllocator(wrapped_object, alloc, ctx) {}

  void free(void* ptr);
};

#ifdef WITH_CUDA
class CudaStorageWeakRefAllocator {
public:
  CudaStorageWeakRefAllocator(PyObject *wrapped_object, THCDeviceAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

  cudaError_t malloc(void** ptr, size_t size, cudaStream_t stream);
  cudaError_t realloc(void** ptr, size_t old_size, size_t size, cudaStream_t stream);
  cudaError_t free(void* ptr);

  THPObjectPtr object;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
};
#endif

#ifdef WITH_NUMPY
class NumpyArrayAllocator: public ObjectPtrAllocator {
public:
  NumpyArrayAllocator(PyObject *wrapped_array):
      ObjectPtrAllocator(wrapped_array) {}

  void* realloc(void* ptr, long size);
  void free(void* ptr);
};
#endif

extern THAllocator THObjectPtrAllocator;
extern THAllocator THStorageWeakRefAllocator;
#ifdef WITH_CUDA
extern THCDeviceAllocator THCStorageWeakRefAllocator;
#endif
#ifdef WITH_NUMPY
extern THAllocator THNumpyArrayAllocator;
#endif

#endif
