#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

bool THPUtils_(parseSlice)(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength)
{
  Py_ssize_t start, stop, step, slicelength;
  if (PySlice_GetIndicesEx(slice, len, &start, &stop, &step, &slicelength) < 0) {
    PyErr_SetString(PyExc_RuntimeError, "Got an invalid slice");
    return false;
  }
  if (step != 1) {
    PyErr_SetString(PyExc_RuntimeError, "Only step == 1 supported");
    return false;
  }
  *ostart = start;
  *ostop = stop;
  if(oslicelength)
    *oslicelength = slicelength;
  return true;
}

bool THPUtils_(parseReal)(PyObject *value, real *result)
{
  if (PyLong_Check(value)) {
    *result = PyLong_AsLongLong(value);
  } else if (PyFloat_Check(value)) {
    *result = PyFloat_AsDouble(value);
  } else {
    // TODO: meaningful error
    PyErr_SetString(PyExc_RuntimeError, "Unrecognized object");
    return false;
  }
  return true;
}

#endif
