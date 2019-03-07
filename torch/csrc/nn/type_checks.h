#pragma once

// Defines type checks and unpacking code for the legacy THNN/THCUNN bindings.
// These checks accept Tensors and Variables.

#include <ATen/ATen.h>

#include <torch/csrc/autograd/python_variable.h>

namespace torch { namespace nn {

inline bool check_type(PyObject* obj, at::TypeID typeID) {
  if (THPVariable_Check(obj)) {
    auto& data_type = ((THPVariable*)obj)->cdata.type();
    return at::globalContext().getNonVariableType(data_type.backend(), data_type.scalarType()).ID() == typeID;
  }
  return false;
}

template<typename T>
inline T* unpack(PyObject* obj) {
  return (T*) ((THPVariable*)obj)->cdata.data().unsafeGetTensorImpl();
}

}} // namespace torch::nn

static inline int get_device(PyObject* args) {
  for (int i = 0, n = PyTuple_GET_SIZE(args); i != n; i++) {
    PyObject* arg = PyTuple_GET_ITEM(args, i);
    if (THPVariable_Check(arg)) {
      auto& tensor = THPVariable_UnpackData(arg);
      if (tensor.is_cuda()) {
        return tensor.get_device();
      }
    }
  }
  return -1;
}

static inline bool THNN_FloatTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CPUFloat);
}

static inline bool THNN_DoubleTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CPUDouble);
}

static inline bool THNN_LongTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CPULong);
}

static inline bool THNN_IntTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CPUInt);
}

static inline THFloatTensor* THNN_FloatTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THFloatTensor>(obj);
}

static inline THDoubleTensor* THNN_DoubleTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THDoubleTensor>(obj);
}

static inline THLongTensor* THNN_LongTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THLongTensor>(obj);
}

static inline THIntTensor* THNN_IntTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THIntTensor>(obj);
}

#ifdef USE_CUDA

static inline bool THNN_CudaHalfTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CUDAHalf);
}

static inline bool THNN_CudaFloatTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CUDAFloat);
}

static inline bool THNN_CudaDoubleTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CUDADouble);
}

static inline bool THNN_CudaLongTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, at::TypeID::CUDALong);
}

static inline THCudaHalfTensor* THNN_CudaHalfTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCudaHalfTensor>(obj);
}

static inline THCudaTensor* THNN_CudaFloatTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCudaTensor>(obj);
}

static inline THCudaDoubleTensor* THNN_CudaDoubleTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCudaDoubleTensor>(obj);
}

static inline THCudaLongTensor* THNN_CudaLongTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCudaLongTensor>(obj);
}

#endif  // USE_CUDA
