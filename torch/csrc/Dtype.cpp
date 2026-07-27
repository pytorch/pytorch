#include <torch/csrc/Dtype.h>

#include <c10/core/ScalarType.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <cstring>

PyObject* THPDtype_New(at::ScalarType scalar_type, const std::string& name) {
  AT_ASSERT(name.length() < DTYPE_NAME_LEN);
  auto type = &THPDtypeType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self)
    throw python_error();
  auto self_ = reinterpret_cast<THPDtype*>(self.get());
  self_->scalar_type = scalar_type;
  std::strncpy(self_->name, name.c_str(), DTYPE_NAME_LEN);
  return self.release();
}

static PyObject* THPDtype_is_floating_point(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isFloatingType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_itemsize(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packUInt64(
      scalarTypeToTypeMeta(self->scalar_type).itemsize());
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_is_complex(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isComplexType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_is_signed(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  if (at::isSignedType(self->scalar_type)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_abbr(THPDtype* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto abbr = c10::getScalarTypeAbbr(self->scalar_type);
  return PyUnicode_FromStringAndSize(
      abbr.data(), static_cast<Py_ssize_t>(abbr.size()));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_reduce(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  /*
   * For singletons, a string is returned. The string should be interpreted
   * as the name of a global variable.
   */
  auto self = reinterpret_cast<THPDtype*>(_self);
  return THPUtils_packString(self->name);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_to_real(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<THPDtype*>(_self);
  auto scalar_type = self->scalar_type;
  if (!at::isFloatingType(self->scalar_type)) {
    scalar_type = at::toRealValueType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDtype_to_complex(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto* self = reinterpret_cast<THPDtype*>(_self);
  auto scalar_type = self->scalar_type;
  if (!at::isComplexType(self->scalar_type)) {
    scalar_type = at::toComplexType(self->scalar_type);
  }
  return Py_NewRef(torch::getTHPDtype(scalar_type));
  END_HANDLE_TH_ERRORS
}

typedef PyObject* (*getter)(PyObject*, void*);

static const std::initializer_list<PyGetSetDef> THPDtype_properties = {
    {"is_floating_point",
     reinterpret_cast<getter>(THPDtype_is_floating_point),
     nullptr,
     nullptr,
     nullptr},
    {"is_complex",
     reinterpret_cast<getter>(THPDtype_is_complex),
     nullptr,
     nullptr,
     nullptr},
    {"is_signed",
     reinterpret_cast<getter>(THPDtype_is_signed),
     nullptr,
     nullptr,
     nullptr},
    {"itemsize",
     reinterpret_cast<getter>(THPDtype_itemsize),
     nullptr,
     nullptr,
     nullptr},
    {"abbr",
     reinterpret_cast<getter>(THPDtype_abbr),
     nullptr,
     nullptr,
     nullptr},
    {nullptr}};

static const std::initializer_list<PyMethodDef> THPDtype_methods = {
    {"__reduce__", THPDtype_reduce, METH_NOARGS, nullptr},
    {"to_real", THPDtype_to_real, METH_NOARGS, nullptr},
    {"to_complex", THPDtype_to_complex, METH_NOARGS, nullptr},
    {nullptr} /* Sentinel */
};

static PyObject* THPDtype_repr(THPDtype* self) {
  return THPUtils_packString(std::string("torch.") + self->name);
}

PyTypeObject THPDtypeType = {
    PyVarObject_HEAD_INIT(nullptr, 0)
    "torch.dtype", /* tp_name */
    sizeof(THPDtype), /* tp_basicsize */
    0, /* tp_itemsize */
    nullptr, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    reinterpret_cast<reprfunc>(THPDtype_repr), /* tp_repr */
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
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyMethodDef*>(std::data(THPDtype_methods)), /* tp_methods */
    nullptr, /* tp_members */
    // NOLINTNEXTLINE(*const-cast)
    const_cast<PyGetSetDef*>(std::data(THPDtype_properties)), /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    nullptr, /* tp_new */
};

void THPDtype_init(PyObject* module) {
  // Set __module__ = "torch" so pickle can find dtype instances without
  // scanning sys.modules. See https://github.com/pytorch/pytorch/issues/65077
  if (PyModule_AddType(module, &THPDtypeType) < 0) {
    throw python_error();
  }
  auto torch_name = THPUtils_packString("torch");
  if (!torch_name)
    throw python_error();
  if (PyDict_SetItemString(THPDtypeType.tp_dict, "__module__", torch_name) <
      0) {
    throw python_error();
  }
  PyType_Modified(&THPDtypeType);
}
