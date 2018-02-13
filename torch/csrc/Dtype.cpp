#include "Dtype.h"

#include <cstring>
#include <structmember.h>
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/tensor_dtypes.h"
#include "torch/csrc/utils/tensor_types.h"
#include "THP.h"

PyObject* THPDtypeClass = nullptr;

PyObject * THPDtype_New(at::Type* cdata, const std::string& name)
{
  auto type = (PyTypeObject*)THPDtypeClass;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->cdata = cdata;
  char *name_cstr = new char[name.length() + 1];
  std::strcpy (name_cstr, name.c_str());
  self_->name = name_cstr;
  return self.release();
}

PyObject *THPDtype_repr(THPDtype *self)
{
  return THPUtils_packString(self->name);
}

static PyObject * THPDtype_element_size(THPDtype *self) {
  return THPUtils_packInt64(self->cdata->elementSizeInBytes());
}

PyObject *THPDtype_get_cdata(THPDtype *self)
{
  HANDLE_TH_ERRORS
  auto& type = self->cdata;
  return PyLong_FromVoidPtr(&type);
  END_HANDLE_TH_ERRORS
}

PyObject *THPDtype_is_cuda(THPDtype *self)
{
  if (self->cdata->is_cuda()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

PyObject *THPDtype_is_sparse(THPDtype *self)
{
  if (self->cdata->is_sparse()) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
}

typedef PyObject *(*getter)(PyObject *, void *);

static PyMethodDef  THPDtype_methods[] = {
  {"element_size",    (PyCFunction)THPDtype_element_size,       METH_NOARGS,  nullptr},
  {nullptr}
};

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
  THPDtype_methods,                      /* tp_methods */
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
  THPDtypeClass = (PyObject*)&THPDtypeType;
  if (PyType_Ready(&THPDtypeType) < 0)
    return false;
  Py_INCREF(&THPDtypeType);
  PyModule_AddObject(module, "dtype", (PyObject *)&THPDtypeType);
  return true;
}
