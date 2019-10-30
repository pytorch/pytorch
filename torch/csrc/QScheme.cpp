#include <torch/csrc/QScheme.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/core/QScheme.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject *THPQScheme_New(at::QScheme qscheme, const std::string& name)
{
  auto type = (PyTypeObject*)&THPQSchemeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPQScheme*>(self.get());
  self_->qscheme = qscheme;
  std::strncpy (self_->name, name.c_str(), QSCHEME_NAME_LEN);
  self_->name[QSCHEME_NAME_LEN] = '\0';
  return self.release();
}

PyObject *THPQScheme_reduce(THPQScheme *self, PyObject *noargs) {
  return THPUtils_packString(self->name);
}

static PyMethodDef THPQScheme_methods[] = {
  {"__reduce__", (PyCFunction)THPQScheme_reduce, METH_NOARGS, nullptr},
  {nullptr}  /* Sentinel */
};

PyObject *THPQScheme_repr(THPQScheme *self)
{
  std::string name = self->name;
  return THPUtils_packString("torch." + name);
}

PyTypeObject THPQSchemeType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.qscheme",                             /* tp_name */
  sizeof(THPQScheme),                          /* tp_basicsize */
  0,                                           /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  nullptr,                                     /* tp_print */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  (reprfunc)THPQScheme_repr,                   /* tp_repr */
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
  THPQScheme_methods,                          /* tp_methods */
  nullptr,                                     /* tp_members */
  nullptr,                                     /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                           /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  nullptr,                                     /* tp_new */
};

void THPQScheme_init(PyObject *module)
{
  if (PyType_Ready(&THPQSchemeType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPQSchemeType);
  if (PyModule_AddObject(module, "qscheme", (PyObject *)&THPQSchemeType) != 0) {
    throw python_error();
  }
}
