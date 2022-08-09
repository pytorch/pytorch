#pragma once
#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/python_headers.h>

namespace torch {
// Sometimes we don't want infinite recursion for subclasses,
// Or a way to achieve the old behaviour.

// This is an internal utility, not exposed to users.
bool torch_function_enabled();
bool should_skip_torch_function();
PyObject* disabled_torch_function_impl();
PyObject* disabled_torch_dispatch_impl();
void set_disabled_torch_function_impl(PyObject* value);
void set_disabled_torch_dispatch_impl(PyObject* value);
// Set ignore_mode to true if you're trying to collect overloaded arguments;
// using mode here will improperly cause you to add ALL objects to the
// overloaded list even if they don't actually have __torch_function__
bool check_has_torch_function(PyObject* obj, bool ignore_mode = false);

bool has_torch_function(PyObject* obj);
bool has_torch_function(c10::ArrayRef<PyObject*> args);

struct TorchFunctionChecker {
  bool has_torch_function(PyObject* obj) {
    return check_has_torch_function(obj);
  }
  bool has_torch_function(c10::ArrayRef<PyObject*> args) {
    for (auto obj : args) {
      if (check_has_torch_function(obj)) {
        return true;
      }
    }
    return false;
  }

 private:
  TorchFunctionChecker() = default;

  template <typename FCheck, typename FHandle, typename FDefault>
  friend auto with_torch_function(FCheck check, FHandle handle, FDefault def);
};

template <typename FCheck, typename FHandle, typename FDefault>
auto with_torch_function(
    FCheck check,
    FHandle handle_torch_function,
    FDefault default_impl) {
  TorchFunctionChecker checker;
  if (!should_skip_torch_function() && check(checker)) {
    return handle_torch_function();
  }
  return default_impl();
}

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
PyObject* THPModule_skip_one_hop_torch_function(PyObject* self, PyObject* args);
PyObject* THPModule_has_torch_function(PyObject*, PyObject* arg);
PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject* obj);
PyObject* THPModule_has_torch_function_variadic(
    PyObject*,
    PyObject* const* args,
    Py_ssize_t nargs);
