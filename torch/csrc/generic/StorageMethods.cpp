static bool THPStorage_(parseSlice)(THPStorage *self, PyObject *slice, Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength)
{
  Py_ssize_t start, stop, step, slicelength, len;
  len = THPStorage_(length)(self);
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

static bool THPStorage_(parseReal)(PyObject *value, real *result)
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

static bool THPStorage_(IsSubclass)(PyObject *storage)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(storage), (PyObject*)&THPStorageType);
}

////////////////////////////////////////////////////////////////////////////////

static PyObject * THPStorage_(size)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(size)(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(elementSize)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THStorage_(elementSize)());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(retain)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStorage_(retain)(self->cdata);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(free)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  THStorage_(free)(self->cdata);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(new)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return THPStorage_(newObject)(THStorage_(new)());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(resize)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  if (!PyLong_Check(number_arg))
    return NULL;
  size_t newsize = PyLong_AsSize_t(number_arg);
  if (PyErr_Occurred())
    return NULL;
  THStorage_(resize)(self->cdata, newsize);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStorage_(fill)(THPStorage *self, PyObject *number_arg)
{
  HANDLE_TH_ERRORS
  real rvalue;
  if (!THPStorage_(parseReal)(number_arg, &rvalue))
    return NULL;
  THStorage_(fill)(self->cdata, rvalue);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

static bool THPDoubleStorage_IsSubclass(PyObject *);
static bool THPFloatStorage_IsSubclass(PyObject *);
static bool THPLongStorage_IsSubclass(PyObject *);
static bool THPIntStorage_IsSubclass(PyObject *);
static bool THPShortStorage_IsSubclass(PyObject *);
static bool THPCharStorage_IsSubclass(PyObject *);
static bool THPByteStorage_IsSubclass(PyObject *);
// TODO: error checking
#define GET_CDATA                                                              \
  PyObject *attr = PyObject_GetAttrString(other_storage, "_cdata");            \
  void *cdata = PyLong_AsVoidPtr(attr);                                        \
  Py_DECREF(attr);
static PyObject * THPStorage_(copy)(THPStorage *self, PyObject *other_storage)
{
  HANDLE_TH_ERRORS
  if (THPDoubleStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyDouble)(self->cdata, (THDoubleStorage*)cdata);
  } else if (THPFloatStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyFloat)(self->cdata, (THFloatStorage*)cdata);
  } else if (THPLongStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyLong)(self->cdata, (THLongStorage*)cdata);
  } else if (THPIntStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyInt)(self->cdata, (THIntStorage*)cdata);
  } else if (THPShortStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyShort)(self->cdata, (THShortStorage*)cdata);
  } else if (THPCharStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyChar)(self->cdata, (THCharStorage*)cdata);
  } else if (THPByteStorage_IsSubclass(other_storage)) {
    GET_CDATA;
    THStorage_(copyByte)(self->cdata, (THByteStorage*)cdata);
  } else {
    // TODO: better error message
    PyErr_SetString(PyExc_RuntimeError, "Copy not implemented for this type");
    return NULL;
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}
#undef GET_CDATA

static PyMethodDef THPStorage_(methods)[] = {
  {"copy", (PyCFunction)THPStorage_(copy), METH_O, NULL},
  {"elementSize", (PyCFunction)THPStorage_(elementSize), METH_NOARGS, NULL},
  {"fill", (PyCFunction)THPStorage_(fill), METH_O, NULL},
  {"free", (PyCFunction)THPStorage_(free), METH_NOARGS, NULL},
  {"new", (PyCFunction)THPStorage_(new), METH_NOARGS, NULL},
  {"resize", (PyCFunction)THPStorage_(resize), METH_O, NULL},
  {"retain", (PyCFunction)THPStorage_(retain), METH_NOARGS, NULL},
  {"size", (PyCFunction)THPStorage_(size), METH_NOARGS, NULL},
  {NULL}
};
