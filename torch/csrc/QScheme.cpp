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
  0,                                           /* tp_dealloc */
  0,                                           /* tp_print */
  0,                                           /* tp_getattr */
  0,                                           /* tp_setattr */
  0,                                           /* tp_reserved */
  (reprfunc)THPQScheme_repr,                   /* tp_repr */
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
  THPQScheme_methods,                          /* tp_methods */
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
