#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.c"
#else

#define GET_SELF THPStorage *self = (THPStorage *)_self

static PyObject *THPStorageClass = NULL;

static void THPStorage_(getClass)()
{
  // TODO: error checking
  if (THPStorageClass)
    return;
  PyObject *torch_module = PyImport_ImportModule("torch");
  PyObject* module_dict = PyModule_GetDict(torch_module);
  THPStorageClass = PyMapping_GetItemString(module_dict, THPStorageStr);
}

PyObject * THPStorage_(newObject)(THStorage *ptr)
{
  // TODO: error checking
  THPStorage_(getClass)();
  PyObject *args = PyTuple_New(0);
  PyObject *kwargs = Py_BuildValue("{s:N}", "cdata", PyLong_FromVoidPtr(ptr));
  PyObject *instance = PyObject_Call(THPStorageClass, args, kwargs);
  Py_DECREF(args);
  Py_DECREF(kwargs);
  return instance;
}

typedef struct {
  PyObject_HEAD
  THStorage *cdata;
} THPStorage;

static void THPStorage_(dealloc)(THPStorage* self)
{
  THStorage_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPStorage_(new)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  static char *keywords[] = {"cdata", NULL};
  PyObject *pointer_obj;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O!", keywords, &PyLong_Type, &pointer_obj))
    return NULL;

  THPStorage *self = (THPStorage *)type->tp_alloc(type, 0);
  if (self != NULL) {
    self->cdata = pointer_obj ? PyLong_AsVoidPtr(pointer_obj) : THLongStorage_new();
    if (self->cdata == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
}

static Py_ssize_t THPStorage_(length)(PyObject *_self)
{
  GET_SELF;
  return THStorage_(size)(self->cdata);
}

static PyObject * THPStorage_(get)(PyObject *_self, PyObject *index)
{
  GET_SELF;
  if (!PyLong_Check(index))
    return NULL;
  size_t nindex = PyLong_AsSize_t(index);
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
  return PyFloat_FromDouble(THStorage_(get)(self->cdata, nindex));
#else
  return PyLong_FromLong(THStorage_(get)(self->cdata, nindex));
#endif
}

static int THPStorage_(set)(PyObject *_self, PyObject *index, PyObject *value)
{
  return -1;
}

static struct PyMemberDef THPStorage_(members)[] = {
  {"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), 0, "C struct pointer"},
  {NULL}
};

static PyMethodDef THPStorage_(methods)[] = {
  {NULL}
};

static PyMappingMethods THPStorage_(mappingmethods) = {
  THPStorage_(length),
  THPStorage_(get),
  THPStorage_(set)
};

static PyTypeObject THPStorageType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch.C." THPStorageBaseStr,          /* tp_name */
  sizeof(THPStorage),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStorage_(dealloc),      /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  0,                                     /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  &THPStorage_(mappingmethods),          /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  0,                                     /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPStorage_(methods),                  /* tp_methods */
  THPStorage_(members),                  /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStorage_(new),                      /* tp_new */
};

bool THPStorage_(init)(PyObject *module)
{
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  return true;
}

#endif
