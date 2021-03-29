#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <structmember.h>

PyTypeObject *THPStreamClass = nullptr;

static PyObject* THPStream_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  uint64_t cdata = 0;
  static char *kwlist[] = {"_cdata", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs, "|K", kwlist, &cdata)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THPStream* self = (THPStream *)ptr.get();
  self->cdata = cdata;
  return (PyObject *)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THPStream_dealloc(THPStream *self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * THPStream_get_device(THPStream *self, void *unused) {
  HANDLE_TH_ERRORS
  return THPDevice_New(c10::Stream::unpack(self->cdata).device());
  END_HANDLE_TH_ERRORS
}

static PyObject * THPStream_eq(THPStream *self, THPStream *other) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(self->cdata == other->cdata);
  END_HANDLE_TH_ERRORS
}

static struct PyMemberDef THPStream_members[] = {
  {(char*)"_cdata",
    T_ULONGLONG, offsetof(THPStream, cdata), READONLY, nullptr},
  {nullptr}
};

static struct PyGetSetDef THPStream_properties[] = {
  {"device", (getter)THPStream_get_device, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPStream_methods[] = {
  {(char*)"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
  {nullptr}
};

PyTypeObject THPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.Stream",                        /* tp_name */
  sizeof(THPStream),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStream_dealloc,         /* tp_dealloc */
  0,                                     /* tp_vectorcall_offset */
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
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  THPStream_methods,                     /* tp_methods */
  THPStream_members,                     /* tp_members */
  THPStream_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPStream_pynew,                       /* tp_new */
};


void THPStream_init(PyObject *module)
{
  THPStreamClass = &THPStreamType;
  Py_TYPE(&THPStreamType) = &PyType_Type;
  if (PyType_Ready(&THPStreamType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPStreamType);
  if (PyModule_AddObject(
      module, "Stream", (PyObject *)&THPStreamType) < 0) {
    throw python_error();
  }
}
