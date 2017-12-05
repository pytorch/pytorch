#include "python_variable_numpy.h"

#ifndef WITH_NUMPY
namespace torch { namespace autograd {
PyObject * THPVariable_numpy(PyObject* pyself, PyObject* arg) {
  return PyErr_Format(PyExc_RuntimeError, "PyTorch was compiled without NumPy support");
}
}}
#else

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_variable.h"

#include <ATen/ATen.h>
#include <memory>
#include <sstream>
#include <stdexcept>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL __numpy_array_api
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

using namespace at;

namespace torch { namespace autograd {

static std::vector<npy_intp> cast_numpy(IntList x) {
  auto nelem = x.size();
  auto result = std::vector<npy_intp>(nelem);
  for (size_t i = 0; i < nelem; i++) {
    result[i] = static_cast<npy_intp>(x[i]);
  }
  return result;
}

static int numpy_dtype(const at::Type& type);

PyObject * THPVariable_numpy(PyObject* pyself, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto& self = reinterpret_cast<THPVariable*>(pyself)->cdata;
  if (self.requires_grad()) {
    throw std::runtime_error(
        "Can't call numpy() on Variable that requires grad. "
        "Use var.detach().numpy() instead.");
  }

  auto dtype = numpy_dtype(self.type());
  auto sizes = cast_numpy(self.sizes());
  auto strides = cast_numpy(self.strides());
  // NumPy strides use bytes. Torch strides use element counts.
  auto element_size_in_bytes = self.type().elementSizeInBytes();
  for (auto& stride : strides) {
    stride *= element_size_in_bytes;
  }

  auto array = THPObjectPtr(PyArray_New(
      &PyArray_Type,
      self.dim(),
      sizes.data(),
      dtype,
      strides.data(),
      self.data_ptr(),
      0,
      NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE,
      nullptr));
  if (!array) return NULL;

  // TODO: This attempts to keep the underlying memory alive by setting the base
  // object of the ndarray to the tensor and disabling resizes on the storage.
  // This is not sufficient. For example, the tensor's storage may be changed
  // via Tensor.set_, which can free the underlying memory.
  Py_INCREF(pyself);
  if (PyArray_SetBaseObject((PyArrayObject*)array.get(), pyself) == -1) {
    return NULL;
  }
  self.storage()->clear_flag(Storage::RESIZABLE);

  return array.release();
  END_HANDLE_TH_ERRORS
}

static int numpy_dtype(const at::Type& type) {
  if (type.is_cuda()) {
    throw std::runtime_error(
        "can't convert CUDA tensor to numpy. Use Tensor.cpu() to "
        "copy the tensor to host memory first.");
  }
  if (type.is_sparse()) {
    throw std::runtime_error(
        "can't convert sparse tensor to numpy. Use Tensor.to_dense() to "
        "convert to a dense tensor first.");
  }
  if (type.backend() == kCPU) {
    switch (type.scalarType()) {
      case kDouble: return NPY_DOUBLE;
      case kFloat: return NPY_FLOAT;
      case kHalf: return NPY_HALF;
      case kLong: return NPY_INT64;
      case kInt: return NPY_INT32;
      case kShort: return NPY_INT16;
      case kByte: return NPY_UINT8;
      default: break;
    }
  }
  std::stringstream ss;
  ss << "NumPy conversion for " << type.toString() << " is not supported";
  throw std::runtime_error(ss.str());
}

}} // namespace torch::autograd

#endif  // WITH_NUMPY
