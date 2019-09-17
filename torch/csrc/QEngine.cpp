#include <torch/csrc/QEngine.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <c10/core/QEngine.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject* THPQEngine_New(at::QEngine qengine, const std::string& name) {
  auto type = (PyTypeObject*)&THPQEngineType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPQEngine*>(self.get());
  self_->qengine = qengine;
  std::strncpy(self_->name, name.c_str(), QENGINE_NAME_LEN);
  self_->name[QENGINE_NAME_LEN] = '\0';
  return self.release();
}

PyObject* THPQEngine_repr(THPQEngine* self) {
  std::string name = self->name;
  return THPUtils_packString("torch." + name);
}

PyTypeObject THPQEngineType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.qengine", /* tp_name */
    sizeof(THPQEngine), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    nullptr, /* tp_print */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    (reprfunc)THPQEngine_repr, /* tp_repr */
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

void THPQEngine_init(PyObject* module) {
  if (PyType_Ready(&THPQEngineType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPQEngineType);
  if (PyModule_AddObject(module, "qengine", (PyObject*)&THPQEngineType) !=
      0) {
    throw python_error();
  }
}
