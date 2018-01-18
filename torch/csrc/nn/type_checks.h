#pragma once

// Defines type checks and unpacking code for the legacy THNN/THCUNN bindings.
// These checks accept Tensors and Variables.

#include <ATen/ATen.h>

#include "THP_API.h"
#include "torch/csrc/autograd/python_variable.h"

namespace torch { namespace nn {

inline bool check_type(PyObject* obj, PyObject* cls, at::TypeID typeID) {
  if ((PyObject*)Py_TYPE(obj) == cls) {
    return true;
  }
  if (THPVariable_Check(obj)) {
    return ((THPVariable*)obj)->cdata.data().type().ID() == typeID;
  }
  return false;
}

template<typename TP, typename T>
inline T* unpack(PyObject* obj, PyObject* cls) {
  if ((PyObject*)Py_TYPE(obj) == cls) {
    return ((TP*)obj)->cdata;
  }
  return (T*) ((THPVariable*)obj)->cdata.data().unsafeGetTH(false);
}

}} // namespace torch::nn


static inline bool THNN_FloatTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THPFloatTensorClass, at::TypeID::CPUFloat);
}

static inline bool THNN_DoubleTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THPDoubleTensorClass, at::TypeID::CPUDouble);
}

static inline bool THNN_LongTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THPLongTensorClass, at::TypeID::CPULong);
}

static inline bool THNN_IntTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THPIntTensorClass, at::TypeID::CPUInt);
}

static inline THFloatTensor* THNN_FloatTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THPFloatTensor, THFloatTensor>(obj, THPFloatTensorClass);
}

static inline THDoubleTensor* THNN_DoubleTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THPDoubleTensor, THDoubleTensor>(obj, THPDoubleTensorClass);
}

static inline THLongTensor* THNN_LongTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THPLongTensor, THLongTensor>(obj, THPLongTensorClass);
}

static inline THIntTensor* THNN_IntTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THPIntTensor, THIntTensor>(obj, THPIntTensorClass);
}

#ifdef WITH_CUDA

static inline bool THNN_CudaHalfTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THCPHalfTensorClass, at::TypeID::CUDAHalf);
}

static inline bool THNN_CudaFloatTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THCPFloatTensorClass, at::TypeID::CUDAFloat);
}

static inline bool THNN_CudaDoubleTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THCPDoubleTensorClass, at::TypeID::CUDADouble);
}

static inline bool THNN_CudaLongTensor_Check(PyObject* obj) {
  return torch::nn::check_type(obj, THCPLongTensorClass, at::TypeID::CUDALong);
}

static inline THCudaHalfTensor* THNN_CudaHalfTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCPHalfTensor, THCudaHalfTensor>(obj, THCPHalfTensorClass);
}

static inline THCudaTensor* THNN_CudaFloatTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCPFloatTensor, THCudaTensor>(obj, THCPFloatTensorClass);
}

static inline THCudaDoubleTensor* THNN_CudaDoubleTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCPDoubleTensor, THCudaDoubleTensor>(obj, THCPDoubleTensorClass);
}

static inline THCudaLongTensor* THNN_CudaLongTensor_Unpack(PyObject* obj) {
  return torch::nn::unpack<THCPLongTensor, THCudaLongTensor>(obj, THCPLongTensorClass);
}

#endif  // WITH_CUDA
