#include <Python.h>
#include "THP.h"

#include "generic/utils.cpp"
#include <TH/THGenerateAllTypes.h>

int THPUtils_getLong(PyObject *index, long *result) {
  if (PyLong_Check(index)) {
    *result = PyLong_AsLong(index);
  } else if (PyInt_Check(index)) {
    *result = PyInt_AsLong(index);
  } else {
    char err_string[512];
    snprintf (err_string, 512, "%s %s",
      "getLong expected int or long, but got type: ",
      index->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return 0;
  }
  return 1;
}

int THPUtils_getCallable(PyObject *arg, PyObject **result) {
  if (!PyCallable_Check(arg))
    return 0;
  *result = arg;
  return 1;
}
