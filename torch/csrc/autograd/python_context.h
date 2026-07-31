#pragma once

#include <ATen/ThreadLocalState.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/object_ptr.h>

#include <memory>
#include <utility>

namespace torch::autograd {

inline void throw_persisted_python_error() {
  python_error err;
  err.persist();
  throw std::move(err);
}

inline std::shared_ptr<c10::SafePyObject> copy_current_py_context() {
  THPObjectPtr context(PyContext_CopyCurrent());
  if (!context) {
    throw_persisted_python_error();
  }
  return std::make_shared<c10::SafePyObject>(
      context.release(), getPyInterpreter());
}

inline THPObjectPtr call_with_context(PyObject* callable, PyObject* args) {
  const auto& context = at::ThreadLocalState::get_python_context();
  if (!context) {
    return THPObjectPtr(PyObject_CallObject(callable, args));
  }
  auto* py_context = context->ptr(getPyInterpreter());
  if (at::ThreadLocalState::is_python_context_origin_thread()) {
    return THPObjectPtr(PyObject_CallObject(callable, args));
  }

  // Context objects cannot be entered concurrently, so give each Python
  // autograd callback its own copy of the backward-launch context.
  THPObjectPtr py_context_copy(PyContext_Copy(py_context));
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
