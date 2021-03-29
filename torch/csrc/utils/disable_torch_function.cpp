#include <torch/csrc/utils/disable_torch_function.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {
  static thread_local bool enable_torch_function = true;
  PyObject* disabled_torch_function = nullptr;

  bool torch_function_enabled() {
      return enable_torch_function;
  }

  PyObject* disabled_torch_function_impl() {
    return disabled_torch_function;
  }

  void set_disabled_torch_function_impl(PyObject* value) {
    disabled_torch_function = value;
  }
}

typedef struct {
    PyObject_HEAD
    /* Type-specific fields go here. */
    bool old_state;
} DisableTorchFunction;

PyObject* DisableTorchFunction__enter(PyObject* self, PyObject *unused) {
    ((DisableTorchFunction*)self)->old_state = torch::enable_torch_function;
    torch::enable_torch_function = false;
    Py_RETURN_NONE;
}

PyObject* DisableTorchFunction__exit(PyObject* self, PyObject *unused) {
    torch::enable_torch_function = ((DisableTorchFunction*)self)->old_state;
    Py_RETURN_NONE;
}

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject *unused) {
    if (torch::enable_torch_function) {
        Py_RETURN_TRUE;
    } else
    {
        Py_RETURN_FALSE;
    }
}

static PyMethodDef DisableTorchFunction_methods[] = { // NOLINT
  {"__enter__",  DisableTorchFunction__enter, METH_NOARGS,  nullptr},
  {"__exit__",   DisableTorchFunction__exit,  METH_VARARGS, nullptr},
  {nullptr, nullptr, 0, nullptr}
};

PyTypeObject DisableTorchFunctionType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C.DisableTorchFunction",             /* tp_name */
  sizeof(DisableTorchFunction),                /* tp_basicsize */
  0,                                           /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                          /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  DisableTorchFunction_methods,                /* tp_methods */
  nullptr,                                     /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  PyType_GenericAlloc,                         /* tp_alloc */
  PyType_GenericNew,                           /* tp_new */
};

PyObject* THPModule_DisableTorchFunctionType() {
  if (PyType_Ready(&DisableTorchFunctionType) < 0) {
      return nullptr;
  }

  return (PyObject *)(&DisableTorchFunctionType);
}

PyObject* THPModule_disable_torch_function(PyObject *self, PyObject *a) {
  HANDLE_TH_ERRORS
  PyObject *func=nullptr, *types=nullptr, *args=nullptr, *kwargs=nullptr;
  if (!PyArg_ParseTuple(a, "OO|OO", &func, &types, &args, &kwargs)) {
    return nullptr;
  }
  py::tuple py_args;
  if (args == nullptr) {
    py_args = py::make_tuple();
  }
  else {
    py_args = py::reinterpret_borrow<py::tuple>(args);
  }

  // These are all C-API calls so no exceptions will be raised
  // and therefore no need for RAII approach to storing
  // the old value.
  bool old_value = torch::enable_torch_function;
  torch::enable_torch_function = false;
  // kwargs can safely be nullptr here.
  PyObject *result = PyObject_Call(func, py_args.ptr(), kwargs);
  torch::enable_torch_function = old_value;
  return result;
  END_HANDLE_TH_ERRORS
}

// Makes sure that we don't check for __torch_function__ on basic Python types
static bool is_basic_python_type(PyTypeObject *tp)
{
  return (
    /* Basic number types */
    tp == &PyBool_Type ||

    tp == &PyLong_Type ||
    tp == &PyFloat_Type ||
    tp == &PyComplex_Type ||

    /* Basic sequence types */
    tp == &PyList_Type ||
    tp == &PyTuple_Type ||
    tp == &PyDict_Type ||
    tp == &PySet_Type ||
    tp == &PyFrozenSet_Type ||
    tp == &PyUnicode_Type ||
    tp == &PyBytes_Type ||

    /* other builtins */
    tp == &PySlice_Type ||
    tp == Py_TYPE(Py_None) ||
    tp == Py_TYPE(Py_Ellipsis) ||
    tp == Py_TYPE(Py_NotImplemented) ||

    PyModule_Check(tp) ||
    /* sentinel to swallow trailing || */
    false
  );
}

inline bool has_torch_function_attr(PyObject* obj) {
  auto attr = PyObject_FastGetAttrString(obj, "__torch_function__");
  return (
    attr.ptr() != nullptr &&
    attr.ptr() != torch::disabled_torch_function);
}

namespace torch {
auto check_has_torch_function(PyObject* obj) -> bool
{
  PyTypeObject *tp = Py_TYPE(obj);
  return (
    !THPVariable_CheckTypeExact(tp) &&
    !is_basic_python_type(tp) &&
    torch::torch_function_enabled() &&
    has_torch_function_attr(obj)
  );
}
} // namespace torch

inline bool sequence_has_torch_function(PyObject* args) {
  Py_ssize_t nargs = PySequence_Fast_GET_SIZE(args);
  for (Py_ssize_t i = 0; i < nargs; i++) {
    PyObject* obj = PySequence_Fast_GET_ITEM(args, i);
    if (torch::check_has_torch_function(obj)) {
      return true;
    }
  }
  return false;
}

inline bool array_has_torch_function(PyObject *const *args, Py_ssize_t nargs) {
  for (Py_ssize_t i = 0; i < nargs; i++) {
    if (torch::check_has_torch_function(args[i])) {
      return true;
    }
  }
  return false;
}

PyObject* THPModule_has_torch_function(PyObject*, PyObject *arg) {
  bool result;  // NOLINT(cppcoreguidelines-init-variables)
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
    result = sequence_has_torch_function(args.ptr());
  }

  if (result) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject *obj) {
  // Special case `THPModule_has_torch_function` for the single arg case.
  if (torch::check_has_torch_function(obj)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject* THPModule_has_torch_function_variadic(PyObject*, PyObject *const *args, Py_ssize_t nargs) {
  if (array_has_torch_function(args, nargs)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}
