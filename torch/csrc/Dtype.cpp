#include <torch/csrc/Dtype.h>

#include <cstring>
#include <structmember.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/csrc/utils/tensor_types.h>

PyObject * THPDtype_New(at::ScalarType scalar_type, const std::string& name)
{
  AT_ASSERT(name.length() < DTYPE_NAME_LEN);
  auto type = (PyTypeObject*)&THPDtypeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->scalar_type = scalar_type;
  std::strncpy(self_->name, name.c_str(), DTYPE_NAME_LEN);
  return self.release();
}

PyObject *THPDtype_is_floating_point(THPDtype *self, PyObject *noargs)
{
  if (at::isFloatingType(self->scalar_type) || at::isComplexType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *THPDtype_reduce(THPDtype *self, PyObject *noargs)
{
  /*
  * For singletons, a string is returned. The string should be interpreted
  * as the name of a global variable.
  */
  return THPUtils_packString(self->name);
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPDtype_properties[] = {
  {"is_floating_point", (getter)THPDtype_is_floating_point, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPDtype_methods[] = {
  {"__reduce__", (PyCFunction)THPDtype_reduce, METH_NOARGS, nullptr},
  {nullptr}  /* Sentinel */
};

PyObject *THPDtype_repr(THPDtype *self)
{
  std::string name = self->name;
  return THPUtils_packString("torch." + name);
}

PyTypeObject THPDtypeType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.dtype",                         /* tp_name */
  sizeof(THPDtype),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  nullptr,                                     /* tp_dealloc */
  nullptr,                                     /* tp_print */
  nullptr,                                     /* tp_getattr */
  nullptr,                                     /* tp_setattr */
  nullptr,                                     /* tp_reserved */
  (reprfunc)THPDtype_repr,               /* tp_repr */
  nullptr,                                     /* tp_as_number */
  nullptr,                                     /* tp_as_sequence */
  nullptr,                                     /* tp_as_mapping */
  nullptr,                                     /* tp_hash  */
  nullptr,                                     /* tp_call */
  nullptr,                                     /* tp_str */
  nullptr,                                     /* tp_getattro */
  nullptr,                                     /* tp_setattro */
  nullptr,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  nullptr,                                     /* tp_traverse */
  nullptr,                                     /* tp_clear */
  nullptr,                                     /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  nullptr,                                     /* tp_iter */
  nullptr,                                     /* tp_iternext */
  THPDtype_methods,                      /* tp_methods */
  nullptr,                                     /* tp_members */
  THPDtype_properties,                   /* tp_getset */
  nullptr,                                     /* tp_base */
  nullptr,                                     /* tp_dict */
  nullptr,                                     /* tp_descr_get */
  nullptr,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  nullptr,                                     /* tp_init */
  nullptr,                                     /* tp_alloc */
  nullptr,                                     /* tp_new */
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
