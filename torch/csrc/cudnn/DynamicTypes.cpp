#include <Python.h>

#include "DynamicTypes.h"

#include "torch/csrc/cuda/THCP.h"

namespace torch { namespace cudnn {

cudnnDataType_t getCudnnDataType(PyObject *tensorClass)
{
  if (tensorClass == THCPFloatTensorClass) {
    return CUDNN_DATA_FLOAT;
  } else if (tensorClass == THCPDoubleTensorClass) {
    return CUDNN_DATA_DOUBLE;
  } else if (tensorClass == THCPHalfTensorClass) {
    return CUDNN_DATA_HALF;
  }
  if (!PyType_Check(tensorClass)) {
    throw std::runtime_error("getCudnnDataType() expects a PyTypeObject");
  }
  std::string msg("getCudnnDataType() not supported for ");
  msg += ((PyTypeObject*)tensorClass)->tp_name;
  throw std::runtime_error(msg);
}

PyObject * getTensorClass(PyObject *args)
{
  for (int i = 0; i < PyTuple_Size(args); i++) {
    PyObject *item = PyTuple_GET_ITEM(args, i);
    if (THPModule_isTensor(item)) {
      return (PyObject*)Py_TYPE(item);
    }
  }
  return NULL;
}

}} // namespace torch::cudnn
