#include "Dtype.h"

#include <cstring>
#include <structmember.h>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_dtypes.h"
#include "torch/csrc/utils/tensor_types.h"

PyObject * THPDtype_New(at::ScalarType scalar_type, const std::string& name)
{
  auto type = (PyTypeObject*)&THPDtypeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->scalar_type = scalar_type;
  std::strncpy (self_->name, name.c_str(), DTYPE_NAME_LEN);
  self_->name[DTYPE_NAME_LEN] = '\0';
  return self.release();
}

PyObject *THPDtype_is_floating_point(THPDtype *self)
{
  if (at::isFloatingType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPDtype_properties[] = {
  {"is_floating_point", (getter)THPDtype_is_floating_point, nullptr, nullptr, nullptr},
  {nullptr}
};

PyObject *THPDtype_repr(THPDtype *self)
{
  return THPUtils_packString(self->name);
}

PyTypeObject THPDtypeType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.dtype",                         /* tp_name */
  sizeof(THPDtype),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDtype_repr,               /* tp_repr */
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
  THPDtype_properties,                   /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  0,                                     /* tp_new */
};

void THPDtype_init(PyObject *module)
{
  if (PyType_Ready(&THPDtypeType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDtypeType);
  if (PyModule_AddObject(module, "dtype", (PyObject *)&THPDtypeType) != 0) {
    throw python_error();
  }
}
