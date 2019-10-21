#include <torch/csrc/MemoryFormat.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/core/MemoryFormat.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject *THPMemoryFormat_New(at::MemoryFormat memory_format, const std::string& name)
{
  auto type = (PyTypeObject*)&THPMemoryFormatType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPMemoryFormat*>(self.get());
  self_->memory_format = memory_format;
  std::strncpy (self_->name, name.c_str(), MEMORY_FORMAT_NAME_LEN);
  self_->name[MEMORY_FORMAT_NAME_LEN] = '\0';
  return self.release();
}

PyObject *THPMemoryFormat_repr(THPMemoryFormat *self)
{
  return THPUtils_packString(self->name);
}

PyTypeObject THPMemoryFormatType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.memory_format",                       /* tp_name */
  sizeof(THPMemoryFormat),                     /* tp_basicsize */
  0,                                           /* tp_itemsize */
  0,                                           /* tp_dealloc */
  0,                                           /* tp_print */
  0,                                           /* tp_getattr */
  0,                                           /* tp_setattr */
  0,                                           /* tp_reserved */
  (reprfunc)THPMemoryFormat_repr,              /* tp_repr */
  0,                                           /* tp_as_number */
  0,                                           /* tp_as_sequence */
  0,                                           /* tp_as_mapping */
  0,                                           /* tp_hash  */
  0,                                           /* tp_call */
  0,                                           /* tp_str */
  0,                                           /* tp_getattro */
  0,                                           /* tp_setattro */
  0,                                           /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                          /* tp_flags */
  0,                                           /* tp_doc */
  0,                                           /* tp_traverse */
  0,                                           /* tp_clear */
  0,                                           /* tp_richcompare */
  0,                                           /* tp_weaklistoffset */
  0,                                           /* tp_iter */
  0,                                           /* tp_iternext */
  0,                                           /* tp_methods */
  0,                                           /* tp_members */
  0,                                           /* tp_getset */
  0,                                           /* tp_base */
  0,                                           /* tp_dict */
  0,                                           /* tp_descr_get */
  0,                                           /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  0,                                           /* tp_init */
  0,                                           /* tp_alloc */
  0,                                           /* tp_new */
};

void THPMemoryFormat_init(PyObject *module)
{
  if (PyType_Ready(&THPMemoryFormatType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPMemoryFormatType);
  if (PyModule_AddObject(module, "memory_format", (PyObject *)&THPMemoryFormatType) != 0) {
    throw python_error();
  }
}
