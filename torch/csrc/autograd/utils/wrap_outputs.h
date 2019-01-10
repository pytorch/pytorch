#pragma once

// Wrap tensor operation outputs as PyObject*

#include <ATen/ATen.h>
#include "torch/csrc/python_headers.h"
#include <tuple>

#include "torch/csrc/Dtype.h"
#include "torch/csrc/Layout.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/python_numbers.h"

namespace torch { namespace autograd { namespace utils {

inline PyObject* wrap(at::Tensor tensor) {
  return THPVariable_Wrap(Variable(std::move(tensor)));
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(2)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(3)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(4)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::move(std::get<3>(tensors))));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(5)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::move(std::get<3>(tensors))));
  PyTuple_SET_ITEM(r.get(), 4, wrap(std::move(std::get<4>(tensors))));
  return r.release();
}

inline PyObject* wrap(at::TensorList tl) {
  auto r = THPObjectPtr{PyTuple_New(tl.size())};
  if (!r) throw python_error();
  for (size_t i = 0; i < tl.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, wrap(tl[i]));
  }
  return r.release();
}

inline PyObject* wrap(bool value) {
  if (value) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

inline PyObject* wrap(int64_t value) {
  return THPUtils_packInt64(value);
}

inline PyObject* wrap(double value) {
  return PyFloat_FromDouble(value);
}

inline PyObject* wrap(void* value) {
  return THPUtils_packInt64(reinterpret_cast<intptr_t>(value));
}

inline PyObject* wrap(at::Scalar scalar) {
  return wrap(scalar.toTensor());
}

inline PyObject* wrap(THPDtype *dtype) {
  Py_INCREF(dtype);
  return (PyObject*)dtype;
}

inline PyObject* wrap(THPLayout *layout) {
  Py_INCREF(layout);
  return (PyObject*)layout;
}

}}} // namespace torch::autograd::utils
