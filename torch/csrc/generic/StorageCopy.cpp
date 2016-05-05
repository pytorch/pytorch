#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/StorageCopy.cpp"
#else

static PyObject * THPStorage_(copy)(THPStorage *self, PyObject *other)
{
  HANDLE_TH_ERRORS
  if (THPDoubleStorage_IsSubclass(other)) {
    THStorage_(copyDouble)(self->cdata, ((THPDoubleStorage*)other)->cdata);
  } else if (THPFloatStorage_IsSubclass(other)) {
    THStorage_(copyFloat)(self->cdata, ((THPFloatStorage*)other)->cdata);
  } else if (THPLongStorage_IsSubclass(other)) {
    THStorage_(copyLong)(self->cdata, ((THPLongStorage*)other)->cdata);
  } else if (THPIntStorage_IsSubclass(other)) {
    THStorage_(copyInt)(self->cdata, ((THPIntStorage*)other)->cdata);
  } else if (THPShortStorage_IsSubclass(other)) {
    THStorage_(copyShort)(self->cdata, ((THPShortStorage*)other)->cdata);
  } else if (THPCharStorage_IsSubclass(other)) {
    THStorage_(copyChar)(self->cdata, ((THPCharStorage*)other)->cdata);
  } else if (THPByteStorage_IsSubclass(other)) {
    THStorage_(copyByte)(self->cdata, ((THPByteStorage*)other)->cdata);
  } else {
    // TODO: better error message
    PyErr_SetString(PyExc_RuntimeError, "Copy not implemented for this type");
    return NULL;
  }
  Py_INCREF(self);
  return (PyObject*)self;
  END_HANDLE_TH_ERRORS
}

#endif
