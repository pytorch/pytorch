#pragma once

// Wrap tensor operation outputs as PyObject*

#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>
#include <torch/csrc/python_headers.h>
#include <initializer_list>
#include <tuple>

#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Layout.h>
#include <torch/csrc/QScheme.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/tensor_qschemes.h>

namespace torch {
namespace autograd {
namespace utils {

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

inline PyObject* wrap(THPDtype* dtype) {
  Py_INCREF(dtype);
  return (PyObject*)dtype;
}

inline PyObject* wrap(at::ScalarType scalarType) {
  return wrap(getTHPDtype(scalarType));
}

inline PyObject* wrap(THPLayout* layout) {
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

inline PyObject* wrap(at::TensorList tl) {
  auto r = THPObjectPtr{PyTuple_New(tl.size())};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(tl.size())) {
    PyTuple_SET_ITEM(r.get(), i, wrap(tl[i]));
  }
  return r.release();
}

inline PyObject* wrap(at::IntArrayRef list) {
  auto r = THPObjectPtr{PyTuple_New(list.size())};
  if (!r)
    throw python_error();
  for (const auto i : c10::irange(list.size())) {
    PyTuple_SET_ITEM(r.get(), i, wrap(list[i]));
  }
  return r.release();
}

inline PyObject* wrap(at::Stream stream) {
  return THPStream_Wrap(stream);
}

namespace detail {
template <typename F, typename Tuple, size_t... Is>
void apply_with_idx_impl(
    const F& f,
    Tuple& t,
    std::index_sequence<Is...> /*indices*/) {
  (void)std::initializer_list<int>{(f(std::get<Is>(t), Is), 0)...};
}

// For tuple(a, b, c), calls f(a, 0), f(b, 1), f(c, 2)
template <typename F, typename... Ts>
void apply_with_idx(const F& f, std::tuple<Ts...>& t) {
  apply_with_idx_impl(f, t, std::index_sequence_for<Ts...>{});
}
} // namespace detail

template <typename... Ts>
PyObject* wrap(std::tuple<Ts...> values) {
  auto r = THPObjectPtr{PyTuple_New(sizeof...(Ts))};
  if (!r)
    throw python_error();
  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyTuple_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);
  return r.release();
}

template <typename... Ts>
PyObject* wrap(PyTypeObject* type, std::tuple<Ts...> values) {
  auto r = THPObjectPtr{PyStructSequence_New(type)};
  if (!r)
    throw python_error();
  detail::apply_with_idx(
      [&](auto& value, size_t idx) {
        PyStructSequence_SET_ITEM(r.get(), idx, wrap(std::move(value)));
      },
      values);
  return r.release();
}

} // namespace utils
} // namespace autograd
} // namespace torch
