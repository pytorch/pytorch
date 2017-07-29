#include "Types.h"
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

cudnnDataType_t getCudnnDataType(const at::Tensor& tensor) {
  if (tensor.type().scalarType() == at::kFloat) {
    return CUDNN_DATA_FLOAT;
  } else if (tensor.type().scalarType() == at::kDouble) {
    return CUDNN_DATA_DOUBLE;
  } else if (tensor.type().scalarType() == at::kHalf) {
    return CUDNN_DATA_HALF;
  }
  std::string msg("getCudnnDataType() not supported for ");
  msg += at::toString(tensor.type().scalarType());
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

void _THVoidTensor_assertContiguous(THVoidTensor *tensor, const std::string& name)
{
  static const std::string error_str = "cuDNN requires contiguous ";
  // Contiguity check
  long long expectedStride = 1;
  for (int i = tensor->nDimension-1; i >= 0; --i) {
    if (tensor->size[i] != 1) {
      if (tensor->stride[i] != expectedStride)
        throw std::invalid_argument(error_str + name);
      expectedStride *= tensor->size[i];
    }
  }
}

}}  // namespace torch::cudnn
