#include <torch/csrc/MemoryFormat.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/core/MemoryFormat.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject* THPMemoryFormat_New(
    at::MemoryFormat memory_format,
    const std::string& name) {
  auto type = &THPMemoryFormatType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPMemoryFormat*>(self.get());
  self_->memory_format = memory_format;
  std::strncpy(self_->name, name.c_str(), MEMORY_FORMAT_NAME_LEN);
  self_->name[MEMORY_FORMAT_NAME_LEN] = '\0';
  return self.release();
}

static PyObject* THPMemoryFormat_repr(THPMemoryFormat* self) {
  return THPUtils_packString(self->name);
}

static PyObject* THPMemoryFormat_reduce(PyObject* _self, PyObject* noargs) {
  auto* self = (THPMemoryFormat*)_self;
  return THPUtils_packString(self->name);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
static PyMethodDef THPMemoryFormat_methods[] = {
    {"__reduce__", THPMemoryFormat_reduce, METH_NOARGS, nullptr},
    {nullptr} /* Sentinel */
};

PyTypeObject THPMemoryFormatType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.memory_format", /* tp_name */
    sizeof(THPMemoryFormat), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPMemoryFormat_repr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THPMemoryFormat_methods, /* tp_methods */
    nullptr, /* tp_members */
    nullptr, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

void THPMemoryFormat_init(PyObject* module) {
  if (PyType_Ready(&THPMemoryFormatType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPMemoryFormatType);
  if (PyModule_AddObject(
          module, "memory_format", (PyObject*)&THPMemoryFormatType) != 0) {
    throw python_error();
  }
}
