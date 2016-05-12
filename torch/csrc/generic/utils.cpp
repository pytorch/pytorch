#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/utils.cpp"
#else

bool THPUtils_(parseSlice)(PyObject *slice, Py_ssize_t len, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength)
{
  Py_ssize_t start, stop, step, slicelength;
  if (PySlice_GetIndicesEx(
// https://bugsfiles.kde.org/attachment.cgi?id=61186
#if PY_VERSION_HEX >= 0x03020000 
			   slice,
#else
			   (PySliceObject *)slice,
#endif
			   len, &start, &stop, &step, &slicelength) < 0) {
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
    *result = (real)PyLong_AsLongLong(value);
  }  else if (PyInt_Check(value)) {
    *result = (real)PyInt_AsLong(value);
  } else if (PyFloat_Check(value)) {
    *result = (real)PyFloat_AsDouble(value);
  } else {
   char err_string[512];
   snprintf (err_string, 512, "%s %s", 
	     "parseReal expected long or float, but got type: ",
	     value->ob_type->tp_name);
    PyErr_SetString(PyExc_RuntimeError, err_string);
    return false;
  }
  return true;
}

bool THPUtils_(checkReal)(PyObject *value)
{
  return PyFloat_Check(value) || PyLong_Check(value) || PyInt_Check(value);
}

PyObject * THPUtils_(newReal)(real value)
{
#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
  return PyFloat_FromDouble(value);
#else
  return PyLong_FromLong(value);
#endif
}

#endif
