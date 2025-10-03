#include <torch/csrc/Layout.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <ATen/Layout.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject* THPLayout_New(at::Layout layout, const std::string& name) {
  auto type = &THPLayoutType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPLayout*>(self.get());
  self_->layout = layout;
  std::strncpy(self_->name, name.c_str(), LAYOUT_NAME_LEN);
  self_->name[LAYOUT_NAME_LEN] = '\0';
  return self.release();
}

static PyObject* THPLayout_repr(THPLayout* self) {
  return THPUtils_packString(self->name);
}

PyTypeObject THPLayoutType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.layout", /* tp_name */
    sizeof(THPLayout), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPLayout_repr, /* tp_repr */
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
    nullptr, /* tp_methods */
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

void THPLayout_init(PyObject* module) {
  if (PyType_Ready(&THPLayoutType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPLayoutType);
  if (PyModule_AddObject(module, "layout", (PyObject*)&THPLayoutType) != 0) {
    throw python_error();
  }
}
