//static bool THPTensor_(IsSubclass)(PyObject *tensor)
//{
//  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
//}

////////////////////////////////////////////////////////////////////////////////

#if !defined(TH_REAL_IS_BYTE) && !defined(TH_REAL_IS_SHORT) && !defined(TH_REAL_IS_CHAR)
static PyObject * THPTensor_(abs)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPTensor *source = self;
  if (!PyArg_ParseTuple(args, "|O!", &THPTensorType, &source))
    return NULL;
  THTensor_(abs)(self->cdata, source->cdata);
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}
#endif

static PyObject * THPTensor_(size)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(size)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newSizeOf)(self->cdata));
  }
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(storage)(THPTensor *self)
{
  HANDLE_TH_ERRORS
  return THPStorage_(newObject)(THTensor_(storage)(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(storageOffset)(THPTensor *self)
{
  HANDLE_TH_ERRORS
  return PyLong_FromLong(THTensor_(storageOffset)(self->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(isSameSizeAs)(THPTensor *self, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPTensor *other;
  if (!PyArg_ParseTuple(args, "O!", &THPTensorType, &other))
    return NULL;
  return PyBool_FromLong(THTensor_(isSameSizeAs)(self->cdata, other->cdata));
  END_HANDLE_TH_ERRORS
}

static PyObject * THPTensor_(stride)(THPTensor *self, PyObject *arg)
{
  HANDLE_TH_ERRORS
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(stride)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newStrideOf)(self->cdata));
  }
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPTensor_(methods)[] = {
#if !defined(TH_REAL_IS_BYTE) && !defined(TH_REAL_IS_SHORT) && !defined(TH_REAL_IS_CHAR)
  {"abs", (PyCFunction)THPTensor_(abs), METH_VARARGS, NULL},
#endif
  {"isSameSizeAs",    (PyCFunction)THPTensor_(isSameSizeAs),    METH_VARARGS, NULL},
  {"size",            (PyCFunction)THPTensor_(size),            METH_VARARGS, NULL},
  {"storage",         (PyCFunction)THPTensor_(storage),         METH_NOARGS,  NULL},
  {"storageOffset",   (PyCFunction)THPTensor_(storageOffset),   METH_NOARGS,  NULL},
  {"stride",          (PyCFunction)THPTensor_(stride),          METH_VARARGS, NULL},
  {NULL}
};
