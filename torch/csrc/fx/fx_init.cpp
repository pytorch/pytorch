#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace fx {

struct ToRestore {
  PyObject* m_self;
  PyMethodDef* m_ml;
#if PY_VERSION_HEX >= 0x03080000
  vectorcallfunc vectorcall;
#endif
  PyObject* original_fn; // The original method we are trying to patch
  PyObject* patch_fn; // The function we're patching in place of original_fn
};

class DecRefGuard {
 public:
  DecRefGuard(PyObject* obj) : obj(obj) {}
  ~DecRefGuard() {
    Py_DECREF(obj);
  }

 private:
  PyObject* obj;
};

PyObject* replacement_method(PyObject* self, PyObject* args, PyObject* kwargs) {
  DecRefGuard self_guard(self);
  // restore the implementation immediately so that patch_fn lives for as little
  // as possible
  ToRestore* to_restore = (ToRestore*)PyBytes_AsString(self);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore->original_fn);
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
  DecRefGuard kwargs_guard(kwargs);

  PyObject* result = nullptr;
  // Creates a tuple of 3 python objects
  PyObject* args_ =
      Py_BuildValue("(OOO)", to_restore->original_fn, args, kwargs);
  if (!args_) {
    return nullptr;
  }
  DecRefGuard args_guard(args_);
  // Calls the patched function with arguments of (original function, args,
  // kwargs)
  result = PyEval_CallObject(to_restore->patch_fn, args_);
  return result;
}
// The general idea is that we're patching a PyCFunctionObject, which has a
// couple relevant parts: m_ml: A PyMethodDef (the actual function to call)
// m_self: The self arg.
// vectorcall: An alternate calling convention (Python 3.8+)
// Usually we call obj.m_ml(obj.m_self, args, kwargs). However, we want to patch
// m_ml with ReplacementMethod (which calls our user-provided `patch_fn`). Thus,
// we also replace `m_self` with `ToRestore`, which contains all the information
// needed to restore the original function.
//
// `patch_function` parses the necessary information from the original
// PyCFunction and then patches it. When that function is called, it calls
// `replacement_method`, which then restores back the original `m_ml` and
// `m_self` values, as well as calling the user-defined `patch_fn`.

static PyObject* patch_function(PyObject* self, PyObject* args) {
  static PyMethodDef ReplacementMethod = {
      "replace",
      (PyCFunction)(void (*)())replacement_method,
      METH_VARARGS | METH_KEYWORDS,
      "Replaced method implementation."};

  ToRestore to_restore = {};
  if (!PyArg_ParseTuple(
          args, "OO", &to_restore.original_fn, &to_restore.patch_fn)) {
    return nullptr;
  }
  if (!PyCFunction_Check(to_restore.original_fn)) {
    std::stringstream err;
    err << "Patched object ";
    PyObject* obj_repr = PyObject_Repr(to_restore.original_fn);
    if (PyUnicode_Check(obj_repr)) {
      err << PyUnicode_AS_DATA(obj_repr) << " ";
    }
    err << " is not a CFunction. Please report a bug to PyTorch!";
    PyErr_SetString(PyExc_RuntimeError, err.str().c_str());
    return nullptr;
  }
  DecRefGuard patch_fn_guard(to_restore.patch_fn);
  Py_INCREF(to_restore.patch_fn);
  DecRefGuard patched_method_guard(to_restore.original_fn);
  Py_INCREF(to_restore.original_fn);
  PyCFunctionObject* patch_method_c =
      ((PyCFunctionObject*)to_restore.original_fn);

  to_restore.m_self = patch_method_c->m_self;
  to_restore.m_ml = patch_method_c->m_ml;
#if PY_VERSION_HEX >= 0x03080000
  to_restore.vectorcall = patch_method_c->vectorcall;
#endif

  patch_method_c->m_self =
      PyBytes_FromStringAndSize((const char*)&to_restore, sizeof(ToRestore));
  patch_method_c->m_ml = &ReplacementMethod;
#if PY_VERSION_HEX >= 0x03080000
  patch_method_c->vectorcall = nullptr;
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
