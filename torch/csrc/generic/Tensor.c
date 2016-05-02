#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.c"
#else

#define GET_SELF THPTensor *self = (THPTensor*)_self;

typedef struct {
  PyObject_HEAD
  THTensor *cdata;
} THPTensor;

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPTensor_(new)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
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

static PyObject * THPTensor_(size)(PyObject *_self)
{
  GET_SELF;
  THLongStorage *size = THTensor_(newSizeOf)(self->cdata);
  return THPLongStorage_newObject(size);
}

static struct PyMemberDef THPTensor_(members)[] = {
  {"_cdata", T_ULONGLONG, offsetof(THPTensor, cdata), 0, "C struct pointer"},
  {NULL}
};

static PyMethodDef THPTensor_(methods)[] = {
  {"size", (PyCFunction)THPTensor_(size), METH_NOARGS, NULL},
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
  THPTensor_(new),                       /* tp_new */
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
