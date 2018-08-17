#include "torch/csrc/Layout.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_strings.h"

#include <ATen/Layout.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject *THPLayout_New(at::Layout layout, const std::string& name)
{
  auto type = (PyTypeObject*)&THPLayoutType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPLayout*>(self.get());
  self_->layout = layout;
  std::strncpy (self_->name, name.c_str(), LAYOUT_NAME_LEN);
  self_->name[LAYOUT_NAME_LEN] = '\0';
  return self.release();
}

PyObject *THPLayout_repr(THPLayout *self)
{
  return THPUtils_packString(self->name);
}

PyTypeObject THPLayoutType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.layout",                        /* tp_name */
  sizeof(THPLayout),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPLayout_repr,              /* tp_repr */
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
  nullptr,                               /* tp_doc */
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
  0,                                     /* tp_new */
};

void THPLayout_init(PyObject *module)
{
  if (PyType_Ready(&THPLayoutType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPLayoutType);
  if (PyModule_AddObject(module, "layout", (PyObject *)&THPLayoutType) != 0) {
    throw python_error();
  }
}
