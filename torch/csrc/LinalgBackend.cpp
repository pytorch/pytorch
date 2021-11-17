#include <torch/csrc/LinalgBackend.h>

#include <ATen/LinalgBackend.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>

#include <structmember.h>
#include <cstring>
#include <string>

PyObject *THPLinalgBackend_New(at::LinalgBackend linalg_backend)
{
  const std::string py_repr = at::LinalgBackendToRepr(linalg_backend);
  auto type = (PyTypeObject*)&THPLinalgBackendType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPLinalgBackend*>(self.get());
  self_->linalg_backend = linalg_backend;
  std::strncpy (self_->name, py_repr.c_str(), LINALG_BACKEND_NAME_LEN);
  self_->name[LINALG_BACKEND_NAME_LEN] = '\0';
  return self.release();
}

PyObject *THPLinalgBackend_repr(THPLinalgBackend *self)
{
  return THPUtils_packString(self->name);
}

PyTypeObject THPLinalgBackendType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.linalg_backend",                       /* tp_name */
  sizeof(THPLinalgBackend),                     /* tp_basicsize */
  0,                                           /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  0,                                           /* tp_vectorcall_offset */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  (reprfunc)THPLinalgBackend_repr,              /* tp_repr */
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
  nullptr,                                     /* tp_new */
};

void THPLinalgBackend_init(PyObject *module)
{
  if (PyType_Ready(&THPLinalgBackendType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPLinalgBackendType);
  if (PyModule_AddObject(module, "linalg_backend", (PyObject *)&THPLinalgBackendType) != 0) {
    throw python_error();
  }
}
