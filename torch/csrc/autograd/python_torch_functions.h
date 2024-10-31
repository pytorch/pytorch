#include <Python.h>

namespace torch::autograd {

extern PyObject* THPVariableFunctionsModule;

// Wrapper converts a raised TypeError into returning NotImplemented
// Used to implement binary arithmetic operators
template <PyObject* (*Func)(PyObject*, PyObject*, PyObject*)>
inline PyObject* TypeError_to_NotImplemented_(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  PyObject* ret = Func(self, args, kwargs);
  if (!ret && PyErr_ExceptionMatches(PyExc_TypeError)) {
    PyErr_Clear();
    Py_INCREF(Py_NotImplemented);
    ret = Py_NotImplemented;
  }
  return ret;
}

void initTorchFunctions();

} // namespace torch::autograd
