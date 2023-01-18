#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

#include <ATen/PythonTorchFunctionTLS.h>

namespace torch {
PyObject* disabled_torch_function = nullptr;
PyObject* disabled_torch_dispatch = nullptr;

bool torch_function_enabled() {
  return at::impl::PythonTorchFunctionTLS::get_disabled_state() ==
      at::impl::TorchFunctionDisabledState::ENABLED;
}

PyObject* disabled_torch_function_impl() {
  return disabled_torch_function;
}

void set_disabled_torch_function_impl(PyObject* value) {
  disabled_torch_function = value;
}

PyObject* disabled_torch_dispatch_impl() {
  return disabled_torch_dispatch;
}

void set_disabled_torch_dispatch_impl(PyObject* value) {
  disabled_torch_dispatch = value;
}
} // namespace torch

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      at::impl::TorchFunctionDisabledState old_state;
} DisableTorchFunctionSubclass;

PyObject* DisableTorchFunctionSubclass__enter(
    PyObject* self,
    PyObject* unused) {
  const auto old_state = at::impl::PythonTorchFunctionTLS::get_disabled_state();
  ((DisableTorchFunctionSubclass*)self)->old_state = old_state;
  if (old_state == at::impl::TorchFunctionDisabledState::ENABLED) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  Py_RETURN_NONE;
}

PyObject* DisableTorchFunctionSubclass__exit(PyObject* self, PyObject* unused) {
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      ((DisableTorchFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject* unused) {
  if (torch::torch_function_enabled()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

static PyMethodDef DisableTorchFunctionSubclass_methods[] = { // NOLINT
    {"__enter__", DisableTorchFunctionSubclass__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableTorchFunctionSubclass__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject DisableTorchFunctionSubclassType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C.DisableTorchFunctionSubclass", /* tp_name */
    sizeof(DisableTorchFunctionSubclass), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    DisableTorchFunctionSubclass_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableTorchFunctionSubclassType() {
  if (PyType_Ready(&DisableTorchFunctionSubclassType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableTorchFunctionSubclassType);
}

typedef struct {
  PyObject_HEAD
      /* Type-specific fields go here. */
      at::impl::TorchFunctionDisabledState old_state;
} DisableTorchFunction;

PyObject* DisableTorchFunction__enter(PyObject* self, PyObject* unused) {
  ((DisableTorchFunctionSubclass*)self)->old_state =
      at::impl::PythonTorchFunctionTLS::get_disabled_state();
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      at::impl::TorchFunctionDisabledState::ALL_DISABLED);
  Py_RETURN_NONE;
}

PyObject* DisableTorchFunction__exit(PyObject* self, PyObject* unused) {
  at::impl::PythonTorchFunctionTLS::set_disabled_state(
      ((DisableTorchFunctionSubclass*)self)->old_state);
  Py_RETURN_NONE;
}

static PyMethodDef DisableTorchFunction_methods[] = { // NOLINT
    {"__enter__", DisableTorchFunction__enter, METH_NOARGS, nullptr},
    {"__exit__", DisableTorchFunction__exit, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject DisableTorchFunctionType = {
    PyVarObject_HEAD_INIT(
        nullptr,
        0) "torch._C.DisableTorchFunction", /* tp_name */
    sizeof(DisableTorchFunction), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    DisableTorchFunction_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    PyType_GenericAlloc, /* tp_alloc */
    PyType_GenericNew, /* tp_new */
};

PyObject* THPModule_DisableTorchFunctionType() {
  if (PyType_Ready(&DisableTorchFunctionType) < 0) {
    return nullptr;
  }

  return (PyObject*)(&DisableTorchFunctionType);
}

PyObject* THPModule_disable_torch_function(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    throw torch::TypeError(
        "expected List or Tuple (got %s)", Py_TYPE(args)->tp_name);
  }

  // These are all C-API calls so no exceptions will be raised
  // and therefore no need for RAII approach to storing
  // the old value.
  auto old_value = at::impl::PythonTorchFunctionTLS::get_disabled_state();
  if (old_value == at::impl::TorchFunctionDisabledState::ENABLED) {
    at::impl::PythonTorchFunctionTLS::set_disabled_state(
        at::impl::TorchFunctionDisabledState::SUBCLASSES_DISABLED);
  }
  // kwargs can safely be nullptr here.
  PyObject* result = PyObject_Call(func, py_args.ptr(), kwargs);
  at::impl::PythonTorchFunctionTLS::set_disabled_state(old_value);
  return result;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModule_disable_torch_dispatch(PyObject* self, PyObject* a) {
  HANDLE_TH_ERRORS
  PyObject *func = nullptr, *types = nullptr, *args = nullptr,
           *kwargs = nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  } else if (PyList_Check(args)) {
    py_args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(args));
  } else if (PyTuple_Check(args)) {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  } else {
    throw torch::TypeError(
        "expected List or Tuple (got %s)", Py_TYPE(args)->tp_name);
  }

  // This implementation is not completely correct.  The moral
  // meaning of this function is that we should do a redispatch
  // "after" PythonKey, aka a redispatch() call.  But we don't have a
  // dispatcher call here; we have an opaque Python object.
  //
  // What we have here is a close approximation: instead of redispatch(), we
  // just exclude Python and all the keys before it, so that we will go
  // to the next key after Python.  The difference, however, is we are
  // now PERMANENTLY after Python.  We don't think there are any legitimate
  // cases where we want to go for another round on the entire dispatcher key
  // set, but if there are, then we will have to do something else here.
  c10::impl::ExcludeDispatchKeyGuard guard_(
      // TODO: add constructor for this specifically
      c10::DispatchKeySet(c10::DispatchKeySet::FULL) -
      c10::DispatchKeySet(
          c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Python)
      // NB: off by one hazard here, but it works out: python key is not
      // included in AFTER, so it is included in the negation (and that's
      // correct: we want to exclude Python key and everything BEFORE it.)
  );
  auto r = PyObject_Call(func, py_args.ptr(), kwargs);
  if (r == nullptr)
    throw python_error();
  return r;
  END_HANDLE_TH_ERRORS
}

// Makes sure that we don't check for __torch_function__ on basic Python types
static bool is_basic_python_type(PyTypeObject* tp) {
  return (
      /* Basic number types */
      tp == &PyBool_Type ||

      tp == &PyLong_Type || tp == &PyFloat_Type || tp == &PyComplex_Type ||

      /* Basic sequence types */
      tp == &PyList_Type || tp == &PyTuple_Type || tp == &PyDict_Type ||
      tp == &PySet_Type || tp == &PyFrozenSet_Type || tp == &PyUnicode_Type ||
      tp == &PyBytes_Type ||

      /* other builtins */
      tp == &PySlice_Type || tp == Py_TYPE(Py_None) ||
      tp == Py_TYPE(Py_Ellipsis) || tp == Py_TYPE(Py_NotImplemented) ||

      PyModule_Check(tp) ||
      /* sentinel to swallow trailing || */
      false);
}

inline bool has_torch_function_attr(PyObject* obj) {
  // NOLINTNEXTLINE(clang-diagnostic-writable-strings)
  auto attr = PyObject_FastGetAttrString(obj, "__torch_function__");
  return (
      attr.ptr() != nullptr && attr.ptr() != torch::disabled_torch_function);
}

namespace torch {
auto check_has_torch_function(PyObject* obj, bool ignore_mode) -> bool {
  if (!ignore_mode && at::impl::torch_function_mode_enabled())
    return true;
  PyTypeObject* tp = Py_TYPE(obj);
  return (
      !THPVariable_CheckTypeExact(tp) && !is_basic_python_type(tp) &&
      torch::torch_function_enabled() && has_torch_function_attr(obj));
}
} // namespace torch

inline bool sequence_has_torch_function(PyObject* args) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  Py_ssize_t nargs = PySequence_Fast_GET_SIZE(args);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(args, i);
    if (torch::check_has_torch_function(obj)) {
      return true;
    }
  }
  return false;
}

inline bool array_has_torch_function(PyObject* const* args, Py_ssize_t nargs) {
  for (Py_ssize_t i = 0; i < nargs; i++) {
    if (torch::check_has_torch_function(args[i])) {
      return true;
    }
  }
  return false;
}

PyObject* THPModule_has_torch_function(PyObject*, PyObject* arg) {
  bool result; // NOLINT(cppcoreguidelines-init-variables)
  if (PyTuple_CheckExact(arg) || PyList_CheckExact(arg)) {
    // Fast path:
    //   If we know that we have a tuple or list, we can skip an INCREF and
    //   DECREF from PySequence_Fast. Core functions will always follow this
    //   convention (almost always tuples), and it shaves ~3.5% off the cost of
    //   the check.
    result = sequence_has_torch_function(arg);
  } else {
    auto args = py::reinterpret_steal<py::object>(
        PySequence_Fast(arg, "expected a sequence"));
    if (!args) {
      return nullptr;
    }
    result = sequence_has_torch_function(args.ptr());
  }

  if (result) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject* obj) {
  // Special case `THPModule_has_torch_function` for the single arg case.
  if (torch::check_has_torch_function(obj)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_variadic(
    PyObject*,
    PyObject* const* args,
    Py_ssize_t nargs) {
  if (array_has_torch_function(args, nargs)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
