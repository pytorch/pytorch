#pragma once

// Wrap tensor operation outputs as PyObject*

#include <ATen/ATen.h>
#include <torch/csrc/python_headers.h>
#include <tuple>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_qschemes.h>
#include <torch/csrc/DynamicTypes.h>

namespace torch { namespace autograd { namespace utils {

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

inline PyObject* wrap(c10::complex<double> value) {
  // I could probably also use FromComplex with a reinterpret cast,
  // but... eh.
  return PyComplex_FromDoubles(value.real(), value.imag());
}

inline PyObject* wrap(void* value) {
  return THPUtils_packInt64(reinterpret_cast<intptr_t>(value));
}

inline PyObject* wrap(THPDtype *dtype) {
  Py_INCREF(dtype);
  return (PyObject*)dtype;
}

inline PyObject* wrap(at::ScalarType scalarType) {
  return wrap(getTHPDtype(scalarType));
}

inline PyObject* wrap(THPLayout *layout) {
  Py_INCREF(layout);
  return (PyObject*)layout;
}

inline PyObject* wrap(at::Layout layout) {
  return wrap(getTHPLayout(layout));
}

inline PyObject* wrap(at::Tensor tensor) {
  return THPVariable_Wrap(Variable(std::move(tensor)));
}

inline PyObject* wrap(const at::Scalar& scalar) {
  return wrap(scalar_to_tensor(scalar));
}

inline PyObject* wrap(at::QScheme qscheme) {
  auto* thp_qscheme = torch::utils::getTHPQScheme(qscheme);
  Py_INCREF(thp_qscheme);
  return thp_qscheme;
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyTuple_New(2)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
  return r.release();
}

inline PyObject* wrap(PyTypeObject *type, std::tuple<at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyStructSequence_New(type)};
  if (!r) throw python_error();
  PyStructSequence_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
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

inline PyObject* wrap(PyTypeObject *type, std::tuple<at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyStructSequence_New(type)};
  if (!r) throw python_error();
  PyStructSequence_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 2, wrap(std::get<2>(tensors)));
  return r.release();
}

inline PyObject* wrap(PyTypeObject *type, std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> tensors) {
  auto r = THPObjectPtr{PyStructSequence_New(type)};
  if (!r) throw python_error();
  PyStructSequence_SET_ITEM(r.get(), 0, wrap(std::get<0>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 1, wrap(std::get<1>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 2, wrap(std::get<2>(tensors)));
  PyStructSequence_SET_ITEM(r.get(), 3, wrap(std::get<3>(tensors)));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor, int64_t> tensors) {
  auto r = THPObjectPtr{PyTuple_New(4)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::get<3>(tensors)));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, float, int64_t> tensors) {
  auto r = THPObjectPtr{PyTuple_New(4)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::move(std::get<3>(tensors))));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t> tensors) {
  auto r = THPObjectPtr{PyTuple_New(5)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::move(std::get<3>(tensors))));
  PyTuple_SET_ITEM(r.get(), 4, wrap(std::get<4>(tensors)));
  return r.release();
}

inline PyObject* wrap(std::tuple<at::Tensor, at::Tensor, float, at::Tensor, int64_t> tensors) {
  auto r = THPObjectPtr{PyTuple_New(5)};
  if (!r) throw python_error();
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 2, wrap(std::move(std::get<2>(tensors))));
  PyTuple_SET_ITEM(r.get(), 3, wrap(std::move(std::get<3>(tensors))));
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 4, wrap(std::move(std::get<4>(tensors))));
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

inline PyObject* wrap(at::IntArrayRef list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r) throw python_error();
  for (size_t i = 0; i < list.size(); ++i) {
    PyTuple_SET_ITEM(r.get(), i, wrap(list[i]));
  }
  return r.release();
}

inline PyObject* wrap(std::tuple<float, int64_t> tensors) {
  auto r = THPObjectPtr{PyTuple_New(2)};
  if (!r) throw python_error();
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 0, wrap(std::move(std::get<0>(tensors))));
  // NOLINTNEXTLINE(performance-move-const-arg)
  PyTuple_SET_ITEM(r.get(), 1, wrap(std::move(std::get<1>(tensors))));
  return r.release();
}

}}} // namespace torch::autograd::utils
