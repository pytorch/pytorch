#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Storage.cpp"
#else

static bool THPStorage_(parseSlice)(THPStorage *self, PyObject *slice,
        Py_ssize_t *ostart, Py_ssize_t *ostop, Py_ssize_t *oslicelength);
static bool THPStorage_(parseReal)(PyObject *, real *);

/* A pointer to RealStorage class defined later in Python */
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

bool THPStorage_(IsSubclass)(PyObject *storage)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(storage), (PyObject*)&THPStorageType);
}

static void THPStorage_(dealloc)(THPStorage* self)
{
  THStorage_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPStorage_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static const char *keywords[] = {"cdata", NULL};
  PyObject *number_arg = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O!", (char **)keywords, &PyLong_Type, &number_arg))
    return NULL;

  THPStorage *self = (THPStorage *)type->tp_alloc(type, 0);
  if (self != NULL) {
    if (kwargs) {
      self->cdata = (THStorage*)PyLong_AsVoidPtr(number_arg);
      THStorage_(retain)(self->cdata);
    } else if (/* !kwargs && */ number_arg) {
      self->cdata = THStorage_(newWithSize)(PyLong_AsLong(number_arg));
    } else {
      self->cdata = THStorage_(new)();
    }

    if (self->cdata == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
  // TODO: cleanup on error
  END_HANDLE_TH_ERRORS
}

static Py_ssize_t THPStorage_(length)(THPStorage *self)
{
  HANDLE_TH_ERRORS
  return THStorage_(size)(self->cdata);
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyObject * THPStorage_(get)(THPStorage *self, PyObject *index)
{
  HANDLE_TH_ERRORS
  /* Integer index */
  if (PyLong_Check(index)) {
    size_t nindex = PyLong_AsSize_t(index);
#if defined(TH_REAL_IS_FLOAT) || defined(TH_REAL_IS_DOUBLE)
    return PyFloat_FromDouble(THStorage_(get)(self->cdata, nindex));
#else
    return PyLong_FromLong(THStorage_(get)(self->cdata, nindex));
#endif
  /* Slice index */
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop, slicelength;
    if (!THPStorage_(parseSlice)(self, index, &start, &stop, &slicelength))
      return NULL;

    real *data = THStorage_(data)(self->cdata);
    real *new_data = (real*)THAlloc(slicelength * sizeof(real));
    // TODO: maybe something faster than memcpy?
    memcpy(new_data, data + start, slicelength * sizeof(real));
    THStorage *new_storage = THStorage_(newWithData)(new_data, slicelength);
    return THPStorage_(newObject)(new_storage);
  }
  PyErr_SetString(PyExc_RuntimeError, "Only indexing with integers and slices supported");
  return NULL;
  END_HANDLE_TH_ERRORS
}

static int THPStorage_(set)(THPStorage *self, PyObject *index, PyObject *value)
{
  HANDLE_TH_ERRORS
  real rvalue;
  if (!THPStorage_(parseReal)(value, &rvalue))
    return -1;

  if (PyLong_Check(index)) {
    THStorage_(set)(self->cdata, PyLong_AsSize_t(index), rvalue);
    return 0;
  } else if (PySlice_Check(index)) {
    Py_ssize_t start, stop;
    if (!THPStorage_(parseSlice)(self, index, &start, &stop, NULL))
      return -1;
    // TODO: check the bounds only once
    for (;start < stop; start++)
      THStorage_(set)(self->cdata, start, rvalue);
    return 0;
  }
  PyErr_SetString(PyExc_RuntimeError, "Only indexing with integers and slices supported at the moment");
  return -1;
  END_HANDLE_TH_ERRORS_RET(-1)
}

static PyMappingMethods THPStorage_(mappingmethods) = {
  (lenfunc)THPStorage_(length),
  (binaryfunc)THPStorage_(get),
  (objobjargproc)THPStorage_(set)
};

PyTypeObject THPStorageType = {
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
  0,   /* will be assigned in init */    /* tp_methods */
  0,   /* will be assigned in init */    /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStorage_(pynew),                    /* tp_new */
};

static struct PyMemberDef THPStorage_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPStorage, cdata), READONLY, NULL},
  {NULL}
};

#include "StorageMethods.cpp"

bool THPStorage_(init)(PyObject *module)
{
  THPStorageType.tp_methods = THPStorage_(methods);
  THPStorageType.tp_members = THPStorage_(members);
  if (PyType_Ready(&THPStorageType) < 0)
    return false;
  Py_INCREF(&THPStorageType);
  PyModule_AddObject(module, THPStorageBaseStr, (PyObject *)&THPStorageType);
  return true;
}

#endif
