#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/csrc/python_headers.h>

namespace torch {
// Sometimes we don't want infinite recursion for subclasses,
// Or a way to achieve the old behaviour.

// This is an internal utility, not exposed to users.
bool torch_function_enabled();
PyObject* disabled_torch_function_impl();
PyObject* disabled_torch_dispatch_impl();
void set_disabled_torch_function_impl(PyObject* value);
void set_disabled_torch_dispatch_impl(PyObject* value);
// Set ignore_mode to true if you're trying to collect overloaded arguments;
// using mode here will improperly cause you to add ALL objects to the
// overloaded list even if they don't actually have __torch_function__
bool check_has_torch_function(PyObject* obj, bool ignore_mode = false);

struct DisableTorchDispatch {
  DisableTorchDispatch()
      : guard_(c10::DispatchKey::Python),
        guard_tls_snapshot_(c10::DispatchKey::PythonTLSSnapshot) {}
  c10::impl::ExcludeDispatchKeyGuard guard_;
  c10::impl::ExcludeDispatchKeyGuard guard_tls_snapshot_;
};

} // namespace torch

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject* unused);
PyObject* THPModule_DisableTorchFunctionType();
PyObject* THPModule_disable_torch_function(PyObject* self, PyObject* args);
PyObject* THPModule_disable_torch_dispatch(PyObject* self, PyObject* args);
PyObject* THPModule_has_torch_function(PyObject*, PyObject* arg);
PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject* obj);
PyObject* THPModule_has_torch_function_variadic(
    PyObject*,
    PyObject* const* args,
    Py_ssize_t nargs);
