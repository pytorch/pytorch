#ifndef THP_CUDNN_TYPES_INC
#define THP_CUDNN_TYPES_INC

#include <Python.h>
#include <cstddef>
#include <cudnn.h>

namespace torch { namespace cudnn {

typedef struct THVoidStorage
{
  void *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  void *allocator;
  void *allocatorContext;
  THVoidStorage *view;
} THVoidStorage;

typedef struct THVoidTensor
{
   long *size;
   long *stride;
   int nDimension;
   THVoidStorage *storage;
   ptrdiff_t storageOffset;
   int refcount;
   char flag;
} THVoidTensor;

struct THPVoidTensor {
  PyObject_HEAD
  THVoidTensor *cdata;
};

PyObject * getTensorClass(PyObject *args);
cudnnDataType_t getCudnnDataType(PyObject *tensorClass);

}}  // namespace torch::cudnn

#endif
