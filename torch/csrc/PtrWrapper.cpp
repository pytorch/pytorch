#include <torch/csrc/python_headers.h>
#include <ATen/Utils.h>
#include <functional>

static PyObject* THPWrapperClass = nullptr;

struct THPWrapper {
  PyObject_HEAD
  void *data;
  void (*destructor)(void*);
};

PyObject * THPWrapper_New(void *data, void (*destructor)(void*))
{
  PyObject *args = PyTuple_New(0);
  if (!args) {
    return nullptr;
  }
  PyObject *result = PyObject_Call(THPWrapperClass, args, nullptr);
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
  wrapper->data = nullptr;
  wrapper->destructor = nullptr;
  return self;
}

static void THPWrapper_dealloc(THPWrapper* self)
{
  self->destructor(self->data);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

PyTypeObject THPWrapperType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch._C._PtrWrapper",                      /* tp_name */
  sizeof(THPWrapper),                          /* tp_basicsize */
  0,                                           /* tp_itemsize */
  (destructor)THPWrapper_dealloc,              /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  nullptr,                                     /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                          /* tp_flags */
  nullptr,                                     /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  nullptr,                                     /* tp_methods */
  nullptr,                                     /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  THPWrapper_pynew,                            /* tp_new */
};

bool THPWrapper_init(PyObject *module)
{
  THPWrapperClass = (PyObject*)&THPWrapperType;
  if (PyType_Ready(&THPWrapperType) < 0)
    return false;
  Py_INCREF(&THPWrapperType);
  return true;
}
