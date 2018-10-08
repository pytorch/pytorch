#include "torch/csrc/TypeInfo.h"

#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/python_numbers.h"
#include "torch/csrc/utils/pybind.h"

#include <ATen/core/FInfo.h>
#include <ATen/core/Error.h>

#include <cstring>
#include <limits>
#include <structmember.h>
#include <sstream>

PyObject *THPFInfo_New(const at::ScalarType& type)
{
  auto finfo = (PyTypeObject*)&THPFInfoType;
  auto self = THPObjectPtr{finfo->tp_alloc(finfo, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = type;
  return self.release();
}

PyObject *THPIInfo_New(const at::ScalarType& type)
{
  auto iinfo = (PyTypeObject*)&THPIInfoType;
  auto self = THPObjectPtr{iinfo->tp_alloc(iinfo, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDTypeInfo*>(self.get());
  self_->type = type;
  return self.release();
}

PyObject *THPDTypeInfo_repr(THPDTypeInfo *self)
{
  std::ostringstream oss;
//  oss << "device(type=\'" << self->device.type() << "\'";
//  if (self->device.has_index()) {
//    oss << ", index=" << self->device.index();
//  }
//  oss << ")";
  oss << "finfo(type=" << self->type << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDTypeInfo_str(THPDTypeInfo *self)
{
  std::ostringstream oss;
  oss << "finfo(type=" << self->type << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPFInfo_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  printf("In THPFInfo_pynew\n");
  //return Py_None;
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "ScalarType(ScalarType type)",
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  AT_CHECK(r.idx == 0, "Not a type");
  at::ScalarType scalar_type = r.scalartype(0);
  return THPFInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

PyObject *THPIInfo_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  printf("In THPIInfo_pynew\n");
  //return Py_None;
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "ScalarType(ScalarType type)",
  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  AT_CHECK(r.idx == 0, "Not a type");
  at::ScalarType scalar_type = r.scalartype(0);
  return THPIInfo_New(scalar_type);
  END_HANDLE_TH_ERRORS
}

static PyObject* THPDTypeInfo_bits(THPDTypeInfo *self, void*)
{
  int bits = elementSize(self->type)*8;
  return PyLong_FromLong(bits);
}

static PyObject* THPFInfo_eps(THPFInfo *self, void*)
{
  switch(self->type) {
  case at::ScalarType::Float:
    return PyFloat_FromDouble(std::numeric_limits<float>::epsilon());
  case at::ScalarType::Double:
    return PyFloat_FromDouble(std::numeric_limits<double>::epsilon());
  case at::ScalarType::Half:
      return PyFloat_FromDouble(std::numeric_limits<at::Half>::epsilon());
  case at::ScalarType::ComplexFloat:
        return PyFloat_FromDouble(std::numeric_limits<float>::epsilon());
      case at::ScalarType::ComplexDouble:
        return PyFloat_FromDouble(std::numeric_limits<double>::epsilon());
  case at::ScalarType::ComplexHalf:
          return PyFloat_FromDouble(std::numeric_limits<at::Half>::epsilon());
  default:
   return Py_NotImplemented;
  }
}

static PyObject* THPFInfo_max(THPFInfo *self, void*)
{
  switch(self->type) {
  case at::ScalarType::Float:
    return PyFloat_FromDouble(std::numeric_limits<float>::max());
  case at::ScalarType::Double:
    return PyFloat_FromDouble(std::numeric_limits<double>::max());
  case at::ScalarType::Half:
      return PyFloat_FromDouble(std::numeric_limits<at::Half>::max());
  case at::ScalarType::ComplexFloat:
        return PyFloat_FromDouble(std::numeric_limits<float>::max());
      case at::ScalarType::ComplexDouble:
        return PyFloat_FromDouble(std::numeric_limits<double>::max());
  case at::ScalarType::ComplexHalf:
          return PyFloat_FromDouble(std::numeric_limits<at::Half>::max());
  default:
   return Py_NotImplemented;
  }
}

static PyObject* THPIInfo_max(THPFInfo *self, void*)
{
  switch(self->type) {
case at::ScalarType::Byte:
  return PyLong_FromLong(std::numeric_limits<unsigned char>::max());
  case at::ScalarType::Char:
    return PyLong_FromLong(std::numeric_limits<char>::max());
      case at::ScalarType::Short:
      return PyLong_FromLong(std::numeric_limits<short>::max());
      break;
      case at::ScalarType::Int:
        return PyLong_FromLong(std::numeric_limits<int>::max());
                case at::ScalarType::Long:
          return PyLong_FromLong (std::numeric_limits<long>::max());
        default:
        return Py_NotImplemented;
}
}

static PyObject* THPFInfo_tiny(THPFInfo *self, void*)
{
  switch(self->type) {
  case at::ScalarType::Float:
    return PyFloat_FromDouble(std::numeric_limits<float>::min());
  case at::ScalarType::Double:
    return PyFloat_FromDouble(std::numeric_limits<double>::min());
    case at::ScalarType::Half:
      return PyFloat_FromDouble(std::numeric_limits<at::Half>::min());
      case at::ScalarType::ComplexFloat:
        return PyFloat_FromDouble(std::numeric_limits<float>::min());
      case at::ScalarType::ComplexDouble:
        return PyFloat_FromDouble(std::numeric_limits<double>::min());
        case at::ScalarType::ComplexHalf:
          return PyFloat_FromDouble(std::numeric_limits<at::Half>::min());
  default:
  return Py_NotImplemented;
}
}



PyObject *THPDTypeInfo_reduce(THPFInfo *self)
{
/*  HANDLE_TH_ERRORS
  auto ret = THPObjectPtr{PyTuple_New(2)};
  if (!ret) throw python_error();

  py::object torch_module = py::module::import("torch");
  py::object torch_device = torch_module.attr("device");
  PyTuple_SET_ITEM(ret.get(), 0, torch_device.release().ptr());

  THPObjectPtr args;
  std::ostringstream oss;
  oss << self->device.type();
  if (self->device.has_index()) {
    args = THPObjectPtr{Py_BuildValue("(si)", oss.str().c_str(), self->device.index())};
  } else {
    args = THPObjectPtr{Py_BuildValue("(s)", oss.str().c_str())};
  }
  if (!args) throw python_error();
  PyTuple_SET_ITEM(ret.get(), 1, args.release());

  return ret.release();
  END_HANDLE_TH_ERRORS
  */
  return Py_None;
}


//typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPFInfo_properties[] = {
  {"bits",       (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
  {"eps",        (getter)THPFInfo_eps, nullptr, nullptr, nullptr},
  {"max",        (getter)THPFInfo_max, nullptr, nullptr, nullptr},
  {"tiny",        (getter)THPFInfo_tiny, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPFInfo_methods[] = {
  //{"__reduce__", (PyCFunction)THPDTypeInfo_reduce, METH_NOARGS, nullptr},
  {nullptr}  /* Sentinel */
};

PyTypeObject THPFInfoType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.finfo",                         /* tp_name */
  sizeof(THPFInfo),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDTypeInfo_repr,               /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPDTypeInfo_str,                /* tp_str */
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
  THPFInfo_methods,                      /* tp_methods */
  0,                                     /* tp_members */
  THPFInfo_properties,                   /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPFInfo_pynew,                    /* tp_new */
};


static struct PyGetSetDef THPIInfo_properties[] = {
  {"bits",       (getter)THPDTypeInfo_bits, nullptr, nullptr, nullptr},
  {"max",        (getter)THPIInfo_max, nullptr, nullptr, nullptr},
  {nullptr}
};

static PyMethodDef THPIInfo_methods[] = {
  //{"__reduce__", (PyCFunction)THPDTypeInfo_reduce, METH_NOARGS, nullptr},
  {nullptr}  /* Sentinel */
};

PyTypeObject THPIInfoType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.iinfo",                         /* tp_name */
  sizeof(THPIInfo),                      /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDTypeInfo_repr,               /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPDTypeInfo_str,                /* tp_str */
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
  THPIInfo_methods,                      /* tp_methods */
  0,                                     /* tp_members */
  THPIInfo_properties,                   /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPIInfo_pynew,                        /* tp_new */
};


void THPDTypeInfo_init(PyObject *module)
{
  printf("Starting to initialize module type_info\n");

  if (PyType_Ready(&THPFInfoType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPFInfoType);
  if (PyModule_AddObject(module, "finfo", (PyObject *)&THPFInfoType) != 0) {
    throw python_error();
  }
  if (PyType_Ready(&THPIInfoType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPIInfoType);
  if (PyModule_AddObject(module, "iinfo", (PyObject *)&THPIInfoType) != 0) {
    throw python_error();
  }
  printf("Done initializing module type_info\n");
}
