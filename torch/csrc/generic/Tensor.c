#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.c"
#else

typedef struct {
  PyObject_HEAD
  THTensor *cdata;
} THPTensor;

////////////////////////////////////////////////////////////////////////////////
// HELPERS
////////////////////////////////////////////////////////////////////////////////

static PyTypeObject THPTensorType;
static bool THPTensor_(IsSubclass)(PyObject *tensor)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
}

////////////////////////////////////////////////////////////////////////////////
// TH WRAPPERS
////////////////////////////////////////////////////////////////////////////////
//
#if !defined(TH_REAL_IS_BYTE) && !defined(TH_REAL_IS_SHORT) && !defined(TH_REAL_IS_CHAR)
static PyObject * THPTensor_(abs)(THPTensor *self, PyObject *args)
{
  THPTensor *source = self;
  if (!PyArg_ParseTuple(args, "|O!", &THPTensorType, &source))
    return NULL;
  THTensor_(abs)(self->cdata, source->cdata);
  Py_INCREF(self);
  return (PyObject*)self;
}
#endif

static PyObject * THPTensor_(size)(THPTensor *self, PyObject *arg)
{
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(size)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newSizeOf)(self->cdata));
  }
}

static PyObject * THPTensor_(storage)(THPTensor *self)
{
  return THPStorage_(newObject)(THTensor_(storage)(self->cdata));
}

static PyObject * THPTensor_(storageOffset)(THPTensor *self)
{
  return PyLong_FromLong(THTensor_(storageOffset)(self->cdata));
}

static PyObject * THPTensor_(isSameSizeAs)(THPTensor *self, PyObject *args)
{
  THPTensor *other;
  if (!PyArg_ParseTuple(args, "O!", &THPTensorType, &other))
    return NULL;
  return PyBool_FromLong(THTensor_(isSameSizeAs)(self->cdata, other->cdata));
}

static PyObject * THPTensor_(stride)(THPTensor *self, PyObject *arg)
{
  int dim = -1;
  if (!PyArg_ParseTuple(arg, "|i", &dim))
    return NULL;

  if (dim != -1) {
    return PyLong_FromLong(THTensor_(stride)(self->cdata, dim));
  } else {
    return THPLongStorage_newObject(THTensor_(newStrideOf)(self->cdata));
  }
}

////////////////////////////////////////////////////////////////////////////////
// PYTHON METHODS
////////////////////////////////////////////////////////////////////////////////

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  long long sizes[] = {-1, -1, -1, -1};
  if (!PyArg_ParseTuple(args, "|LLLL", &sizes[0], &sizes[1], &sizes[2], &sizes[3]))
    return NULL;

  THPTensor *self = (THPTensor *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->cdata = THTensor_(newWithSize4d)(sizes[0], sizes[1], sizes[2], sizes[3]);
    if (self->cdata == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
}

////////////////////////////////////////////////////////////////////////////////
// PYTHON DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

static struct PyMemberDef THPTensor_(members)[] = {
  {"_cdata", T_ULONGLONG, offsetof(THPTensor, cdata), READONLY, "C struct pointer"},
  {NULL}
};

static PyMethodDef THPTensor_(methods)[] = {
#if !defined(TH_REAL_IS_BYTE) && !defined(TH_REAL_IS_SHORT) && !defined(TH_REAL_IS_CHAR)
  {"abs", (PyCFunction)THPTensor_(abs), METH_VARARGS, NULL},
#endif
  {"isSameSizeAs", (PyCFunction)THPTensor_(isSameSizeAs), METH_VARARGS, NULL},
  {"size", (PyCFunction)THPTensor_(size), METH_VARARGS, NULL},
  {"storage", (PyCFunction)THPTensor_(storage), METH_NOARGS, NULL},
  {"storageOffset", (PyCFunction)THPTensor_(storageOffset), METH_NOARGS, NULL},
  {"stride", (PyCFunction)THPTensor_(stride), METH_VARARGS, NULL},
  {NULL}
};


static PyTypeObject THPTensorType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch.C." THPTensorBaseStr,           /* tp_name */
  sizeof(THPTensor),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPTensor_(dealloc),       /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPTensor_(methods),                   /* tp_methods */
  THPTensor_(members),                   /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPTensor_(pynew),                     /* tp_new */
};

bool THPTensor_(init)(PyObject *module)
{
  if (PyType_Ready(&THPTensorType) < 0)
    return false;
  Py_INCREF(&THPTensorType);
  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  return true;
}

#endif
