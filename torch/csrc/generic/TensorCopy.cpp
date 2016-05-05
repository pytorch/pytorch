#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/TensorCopy.cpp"
#else

static PyObject * THPTensor_(copy)(THPTensor *self, PyObject *other)
{
  HANDLE_TH_ERRORS
  if (THPDoubleTensor_IsSubclass(other)) {
    THTensor_(copyDouble)(self->cdata, ((THPDoubleTensor*)other)->cdata);
  } else if (THPFloatTensor_IsSubclass(other)) {
    THTensor_(copyFloat)(self->cdata, ((THPFloatTensor*)other)->cdata);
  } else if (THPLongTensor_IsSubclass(other)) {
    THTensor_(copyLong)(self->cdata, ((THPLongTensor*)other)->cdata);
  } else if (THPIntTensor_IsSubclass(other)) {
    THTensor_(copyInt)(self->cdata, ((THPIntTensor*)other)->cdata);
  } else if (THPShortTensor_IsSubclass(other)) {
    THTensor_(copyShort)(self->cdata, ((THPShortTensor*)other)->cdata);
  } else if (THPCharTensor_IsSubclass(other)) {
    THTensor_(copyChar)(self->cdata, ((THPCharTensor*)other)->cdata);
  } else if (THPByteTensor_IsSubclass(other)) {
    THTensor_(copyByte)(self->cdata, ((THPByteTensor*)other)->cdata);
  } else {
    // TODO: better error message
    PyErr_SetString(PyExc_RuntimeError, Py_TYPE(other)->tp_name);
    return NULL;
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#endif

