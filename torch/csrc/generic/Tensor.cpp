#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tensor.cpp"
#else

bool THPTensor_(IsSubclass)(PyObject *tensor)
{
  return PyObject_IsSubclass((PyObject*)Py_TYPE(tensor), (PyObject*)&THPTensorType);
}

static void THPTensor_(dealloc)(THPTensor* self)
{
  THTensor_(free)(self->cdata);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPTensor_(pynew)(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  THPLongStorage *storage_obj = NULL;
  long long sizes[] = {-1, -1, -1, -1};
  // Check if it's a long storage
  if (PyTuple_Size(args) == 1) {
    PyObject *arg = PyTuple_GetItem(args, 0);
    if (THPLongStorage_IsSubclass(arg)) {
      storage_obj = (THPLongStorage*)arg;
    }
  }
  // If not, try to parse integers
#define ERRMSG ";Expected torch.LongStorage or up to 4 integers as arguments"
  if (!storage_obj && !PyArg_ParseTuple(args, "|LLLL" ERRMSG, &sizes[0], &sizes[1], &sizes[2], &sizes[3]))
    return NULL;

  THPTensor *self = (THPTensor *)type->tp_alloc(type, 0);
  if (self != NULL) {
    if (storage_obj)
        self->cdata = THTensor_(newWithSize)(storage_obj->cdata, NULL);
    else
        self->cdata = THTensor_(newWithSize4d)(sizes[0], sizes[1], sizes[2], sizes[3]);
    if (self->cdata == NULL) {
      Py_DECREF(self);
      return NULL;
    }
  }
  return (PyObject *)self;
  // TODO: cleanup on error
  END_HANDLE_TH_ERRORS
}

PyTypeObject THPTensorType = {
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
  THPTensor_(pynew),                     /* tp_new */
};

static struct PyMemberDef THPTensor_(members)[] = {
  {(char*)"_cdata", T_ULONGLONG, offsetof(THPTensor, cdata), READONLY, NULL},
  {NULL}
};

#include "TensorMethods.cpp"

bool THPTensor_(init)(PyObject *module)
{
  THPTensorType.tp_methods = THPTensor_(methods);
  THPTensorType.tp_members = THPTensor_(members);
  if (PyType_Ready(&THPTensorType) < 0)
    return false;
  Py_INCREF(&THPTensorType);
  PyModule_AddObject(module, THPTensorBaseStr, (PyObject *)&THPTensorType);
  return true;
}

#endif
