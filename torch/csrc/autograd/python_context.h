#pragma once

#include <ATen/ThreadLocalPythonObjects.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>

#include <cstdint>
#include <utility>

namespace torch::autograd {

inline void throw_persisted_python_error() {
  python_error err;
  err.persist();
  throw std::move(err);
}

inline bool is_context_origin_thread() {
  if (!at::impl::ThreadLocalPythonObjects::contains(
          "context_origin_thread_id")) {
    return false;
  }

  auto origin_thread_id =
      at::impl::ThreadLocalPythonObjects::get("context_origin_thread_id");
  auto* py_origin_thread_id = origin_thread_id->ptr(getPyInterpreter());
  if (Py_IsNone(py_origin_thread_id)) {
    return false;
  }

  uint64_t origin_id = 0;
  try {
    origin_id = THPUtils_unpackUInt64(py_origin_thread_id);
  } catch (python_error& err) {
    err.persist();
    throw;
  }
  return origin_id == static_cast<uint64_t>(PyThread_get_thread_ident());
}

inline THPObjectPtr call_with_context(PyObject* callable, PyObject* args) {
  if (!at::impl::ThreadLocalPythonObjects::contains("context")) {
    return THPObjectPtr(PyObject_CallObject(callable, args));
  }

  auto context = at::impl::ThreadLocalPythonObjects::get("context");
  auto* py_context = context->ptr(getPyInterpreter());
  if (Py_IsNone(py_context)) {
    return THPObjectPtr(PyObject_CallObject(callable, args));
  }
  if (is_context_origin_thread()) {
    return THPObjectPtr(PyObject_CallObject(callable, args));
  }

  // Context objects cannot be entered concurrently, so give each Python
  // autograd callback its own copy of the backward-launch context.
  THPObjectPtr copy_fn(PyObject_GetAttrString(py_context, "copy"));
  if (!copy_fn) {
    throw_persisted_python_error();
  }
  THPObjectPtr py_context_copy(PyObject_CallNoArgs(copy_fn));
  if (!py_context_copy) {
    throw_persisted_python_error();
  }
  THPObjectPtr run_fn(PyObject_GetAttrString(py_context_copy, "run"));
  if (!run_fn) {
    throw_persisted_python_error();
  }

  auto num_args = PyTuple_GET_SIZE(args);
  THPObjectPtr context_args(PyTuple_New(num_args + 1));
  if (!context_args) {
    throw_persisted_python_error();
  }
  Py_INCREF(callable);
  PyTuple_SET_ITEM(context_args.get(), 0, callable);
  for (Py_ssize_t i = 0; i < num_args; i++) {
    PyObject* item = PyTuple_GET_ITEM(args, i);
    Py_INCREF(item);
    PyTuple_SET_ITEM(context_args.get(), i + 1, item);
  }

  return THPObjectPtr(PyObject_CallObject(run_fn, context_args.get()));
}

} // namespace torch::autograd
