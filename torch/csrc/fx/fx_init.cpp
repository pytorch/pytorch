#include <torch/csrc/utils/pybind.h>
#include <iostream>

namespace torch {
namespace fx {

struct ToRestore {
  PyObject* m_self;
  PyMethodDef* m_ml;
#if PY_VERSION_HEX >= 0x03080000
  vectorcallfunc vectorcall;
#endif
  PyObject* patched_method;
  PyObject* patch_fn;
};

PyObject* replacement_method(PyObject* self, PyObject* args, PyObject* kwargs) {
  // restore the implementation immediately so that patch_fn lives for as little
  // as possible
  ToRestore* to_restore = (ToRestore*)PyBytes_AsString(self);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore->patched_method);
  patch_method_c->m_self = to_restore->m_self;
  patch_method_c->m_ml = to_restore->m_ml;
#if PY_VERSION_HEX >= 0x03080000
  patch_method_c->vectorcall = to_restore->vectorcall;
#endif

  if (kwargs) {
    Py_INCREF(kwargs);
  } else {
    kwargs = PyDict_New();
  }

  PyObject* result = nullptr;
  PyObject* args_ =
      Py_BuildValue("(OOO)", to_restore->patched_method, args, kwargs);
  if (!args_) {
    goto exit;  // NOLINT
  }
  result = PyEval_CallObject(to_restore->patch_fn, args_);
  Py_DECREF(args_);
  if (!result) {
    goto exit; // NOLINT
  }

exit:
  Py_DECREF(to_restore->patched_method);
  Py_DECREF(to_restore->patch_fn);
  Py_DECREF(self);
  Py_DECREF(kwargs);
  return result;
}


static PyObject* patch_function(PyObject* self, PyObject* args) {
  static PyMethodDef ReplacementMethod = {
      "replace",
      (PyCFunction)(void (*)())replacement_method,
      METH_VARARGS | METH_KEYWORDS,
      "Replaced method implementation."};

  ToRestore to_restore = {};
  if (!PyArg_ParseTuple(
          args, "OO", &to_restore.patched_method, &to_restore.patch_fn)) {
    return nullptr;
  }
  if (!PyCFunction_Check(to_restore.patched_method)) {
    PyErr_SetString(PyExc_RuntimeError, "not a CFunction");
    return nullptr;
  }
  Py_INCREF(to_restore.patch_fn);
  Py_INCREF(to_restore.patched_method);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore.patched_method);

  to_restore.m_self = patch_method_c->m_self;
  to_restore.m_ml = patch_method_c->m_ml;
#if PY_VERSION_HEX >= 0x03080000
  to_restore.vectorcall = patch_method_c->vectorcall;
#endif

  patch_method_c->m_self =
      PyBytes_FromStringAndSize((const char*)&to_restore, sizeof(ToRestore));
  patch_method_c->m_ml = &ReplacementMethod;
#if PY_VERSION_HEX >= 0x03080000
    patch_method_c->vectorcall = NULL;
#endif
  return Py_None;
}


void initFx(PyObject* module) {
  static std::array<PyMethodDef, 2> PatchMethods = {{
      {"patch_function", patch_function, METH_VARARGS, "Save"},
      {nullptr},
  }};

  static struct PyModuleDef path = {
      PyModuleDef_HEAD_INIT,
      "patch", /* name of module */
      "", /* module documentation, may be NULL */
      -1, /* size of per-interpreter state of the module, or -1 if the module
            keeps state in global variables. */
      PatchMethods.data()};
  PyObject* patch = PyModule_Create(&path);
  if (!patch) {
    throw python_error();
  }
  if (PyModule_AddObject(module, "_fx", patch) != 0) {
    throw python_error();
  }
}
} // namespace fx
} // namespace torch