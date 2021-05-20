#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <structmember.h>

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject *THPStreamClass = nullptr;

static PyObject* THPStream_pynew(
  PyTypeObject *type, PyObject *args, PyObject *kwargs) {
  HANDLE_TH_ERRORS
  uint64_t cdata = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,clang-diagnostic-writable-strings)
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

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyMemberDef THPStream_members[] = {
  {(char*)"_cdata",
    T_ULONGLONG, offsetof(THPStream, cdata), READONLY, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static struct PyGetSetDef THPStream_properties[] = {
  {"device", (getter)THPStream_get_device, nullptr, nullptr, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables)
static PyMethodDef THPStream_methods[] = {
  {(char*)"__eq__", (PyCFunction)THPStream_eq, METH_O, nullptr},
  {nullptr}
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PyTypeObject THPStreamType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.Stream",                        /* tp_name */
  sizeof(THPStream),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  (destructor)THPStream_dealloc,         /* tp_dealloc */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_vectorcall_offset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_getattr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_setattr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_reserved */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_repr */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_number */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_sequence */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_mapping */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_hash  */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_call */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_str */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_getattro */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_setattro */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  nullptr,                                  /* tp_doc */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_traverse */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_clear */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_iter */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_iternext */
  THPStream_methods,                     /* tp_methods */
  THPStream_members,                     /* tp_members */
  THPStream_properties,                  /* tp_getset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_base */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_dict */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_descr_get */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  // NOLINTNEXTLINE(modernize-use-nullptr)
  0,                                     /* tp_init */
  // NOLINTNEXTLINE(modernize-use-nullptr)
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
