#include "Dtype.h"

#include <cstring>
#include <structmember.h>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_dtypes.h"
#include "torch/csrc/utils/tensor_types.h"

PyObject * THPDtype_New(at::Type* cdata, const std::string& name, bool is_cuda, bool is_sparse)
{
  auto type = (PyTypeObject*)&THPDtypeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->cdata = cdata;
  std::strncpy (self_->name, name.c_str(), DTYPE_NAME_LEN);
  self_->name[DTYPE_NAME_LEN] = '\0';
  self_->is_cuda = is_cuda;
  self_->is_sparse = is_sparse;
  return self.release();
}

PyObject *THPDtype_repr(THPDtype *self)
{
  return THPUtils_packString(self->name);
}

PyObject *THPDtype_get_cdata(THPDtype *self)
{
  return PyLong_FromVoidPtr(self->cdata);
}

PyObject *THPDtype_is_cuda(THPDtype *self)
{
  if (self->is_cuda) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *THPDtype_is_sparse(THPDtype *self)
{
  if (self->is_sparse) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPDtype_properties[] = {
  {"_cdata",       (getter)THPDtype_get_cdata, nullptr, nullptr, nullptr},
  {"is_cuda",      (getter)THPDtype_is_cuda, nullptr, nullptr, nullptr},
  {"is_sparse",    (getter)THPDtype_is_sparse, nullptr, nullptr, nullptr},
  {nullptr}
};

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

bool THPDtype_init(PyObject *module)
{
  if (PyType_Ready(&THPDtypeType) < 0)
    return false;
  Py_INCREF(&THPDtypeType);
  PyModule_AddObject(module, "dtype", (PyObject *)&THPDtypeType);
  return true;
}
