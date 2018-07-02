#pragma once

#include "torch/csrc/python_headers.h"
#include <type_traits>
#include <memory>

#include <TH/TH.h>
#ifdef USE_CUDA
#include <THC/THC.h>
#endif

#include "torch/csrc/utils/object_ptr.h"

// Adapted from fblualib
class ObjectPtrAllocator {
public:
  ObjectPtrAllocator(PyObject *wrapped_object):
      ObjectPtrAllocator(wrapped_object, getTHDefaultAllocator(), nullptr) {}

  ObjectPtrAllocator(PyObject *wrapped_object, THAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

  void* malloc(ptrdiff_t size);
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

#ifdef USE_CUDA
class CudaStorageWeakRefAllocator {
public:
  CudaStorageWeakRefAllocator(PyObject *wrapped_object, THCDeviceAllocator *alloc, void *ctx) {
    Py_XINCREF(wrapped_object);
    object = wrapped_object;
    allocator = alloc;
    allocatorContext = ctx;
  }

  void* malloc(size_t size);
  void free(void* ptr);

  THPObjectPtr object;
  THCDeviceAllocator *allocator;
  void *allocatorContext;
};
#endif

at::Allocator* getTHObjectPtrAllocator();
at::Allocator* getTHStorageWeakRefAllocator();
#ifdef USE_CUDA
at::Allocator* getTHCStorageWeakRefAllocator();
#endif
