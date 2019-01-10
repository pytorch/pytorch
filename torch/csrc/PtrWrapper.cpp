#include "torch/csrc/python_headers.h"
#include <functional>

static PyObject* THPWrapperClass = NULL;

struct THPWrapper {
  PyObject_HEAD
  void *data;
  void (*destructor)(void*);
};

PyObject * THPWrapper_New(void *data, void (*destructor)(void*))
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    return NULL;
  }
  PyObject *result = PyObject_Call(THPWrapperClass, args, NULL);
  if (result) {
    THPWrapper* wrapper = (THPWrapper*) result;
    wrapper->data = data;
    wrapper->destructor = destructor;
  }
  Py_DECREF(args);
  return result;
}

bool THPWrapper_check(PyObject * obj)
{
  return (PyObject*)Py_TYPE(obj) == THPWrapperClass;
}

void * THPWrapper_get(PyObject * obj)
{
  return ((THPWrapper*)obj)->data;
}

static PyObject * THPWrapper_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  PyObject* self = type->tp_alloc(type, 0);
  THPWrapper* wrapper = (THPWrapper*) self;
  wrapper->data = NULL;
  wrapper->destructor = NULL;
  return self;
}

static void THPWrapper_dealloc(THPWrapper* self)
{
  self->destructor(self->data);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject THPWrapperType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "torch._C._PtrWrapper",                /* tp_name */
  sizeof(THPWrapper),                    /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPWrapper_dealloc,        /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  NULL,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  0,                                     /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPWrapper_pynew,                      /* tp_new */
};

bool THPWrapper_init(PyObject *module)
{
  THPWrapperClass = (PyObject*)&THPWrapperType;
  if (PyType_Ready(&THPWrapperType) < 0)
    return false;
  Py_INCREF(&THPWrapperType);
  return true;
}
