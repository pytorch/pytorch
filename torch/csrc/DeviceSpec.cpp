#include "DeviceSpec.h"

#include <cstring>
#include <structmember.h>
#include <sstream>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_strings.h"

PyObject *THPDeviceSpec_New(torch::DeviceType device_type, int64_t device_index, bool is_default)
{
  auto type = (PyTypeObject*)&THPDeviceSpecType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDeviceSpec*>(self.get());
  self_->device_type = device_type;
  self_->device_index = device_index;
  self_->is_default = is_default;
  return self.release();
}

static inline std::string deviceTypeString(torch::DeviceType device_type) {
  switch (device_type) {
    case torch::DeviceType::CUDA:
      return "cuda";
    case torch::DeviceType::CPU:
      return "cpu";
    default:
      throw std::runtime_error("unexpected device type");
  }
}

PyObject *THPDeviceSpec_repr(THPDeviceSpec *self)
{
  std::ostringstream oss;
  oss << "DeviceSpec(device_type=\'" << deviceTypeString(self->device_type) << "\'";
  if (!self->is_default) {
    oss << ", device_index=" << self->device_index;
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDeviceSpec_str(THPDeviceSpec *self)
{
  std::ostringstream oss;
  if (!self->is_default) {
    oss << deviceTypeString(self->device_type) << ":" << self->device_index;
  } else {
    oss << deviceTypeString(self->device_type);
  }
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDeviceSpec_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "DeviceSpec(Device device)",
    "DeviceSpec(String device_type, int64_t? device_index=-1)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto device = r.device(0);
    return THPDeviceSpec_New(device.device_type, device.device_index, device.is_default);
  } else if (r.idx == 1) {
    auto as_device = r.device(0);  // this works, because device can take strings
    auto device_type = r.string(0);
    if (!as_device.is_default) {
      throw std::runtime_error("device_type must not include index because index is passed explicitly " + device_type);
    }

    auto is_default = r.isNone(1);
    auto device_index = r.toInt64WithDefault(1, -1);
    // make sure this is constructible
    auto device = torch::Device(as_device.device_type, device_index, is_default);
    return THPDeviceSpec_New(device.device_type, device.device_index, device.is_default);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_type(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(deviceTypeString(self->device_type).c_str());
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_index(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  if (self->is_default) {
    Py_RETURN_NONE;
  } else {
    return THPUtils_packInt64(self->device_index);
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPDeviceSpec_cuda_index(THPDeviceSpec *self)
{
  HANDLE_TH_ERRORS
  if (self->device_type == torch::DeviceType::CUDA) {
    return THPUtils_packInt64(self->device_index);
  }
  std::ostringstream oss;
  oss << "cuda_index only supported on cuda device, got: " << deviceTypeString(self->device_type);
  throw std::runtime_error(oss.str());
  END_HANDLE_TH_ERRORS
}

static bool THPDeviceSpec_equal(THPDeviceSpec *a, THPDeviceSpec *b) {
  return a->device_type == b->device_type && a->device_index == b->device_index && a->is_default == b->is_default;
}

PyObject *THPDeviceSpec_rc(PyObject *a, PyObject *b, int op) {
  HANDLE_TH_ERRORS
  if (!THPDeviceSpec_Check(a)) {
    throw torch::TypeError("DeviceSpec can only be compared with DeviceType, got %s", Py_TYPE(a)->tp_name);
  }
  if (!THPDeviceSpec_Check(b)) {
    throw torch::TypeError("DeviceSpec can only be compared with DeviceType, got %s", Py_TYPE(b)->tp_name);
  }
  THPDeviceSpec *da = reinterpret_cast<THPDeviceSpec*>(a);
  THPDeviceSpec *db = reinterpret_cast<THPDeviceSpec*>(b);

  switch(op) {
    case Py_EQ:
      if (THPDeviceSpec_equal(da, db)) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      if (!THPDeviceSpec_equal(da, db)) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_LT:
    case Py_LE:
    case Py_GT:
    case Py_GE:
      throw torch::TypeError("comparison not implemented");
    default:
      throw torch::TypeError("unexpected comparison op");
  }
  END_HANDLE_TH_ERRORS
}

typedef PyObject *(*getter)(PyObject *, void *);

static struct PyGetSetDef THPDeviceSpec_properties[] = {
  {"type",       (getter)THPDeviceSpec_type, nullptr, nullptr, nullptr},
  {"index",      (getter)THPDeviceSpec_index, nullptr, nullptr, nullptr},
  {"cuda_index", (getter)THPDeviceSpec_cuda_index, nullptr, nullptr, nullptr},
  {nullptr}
};

PyTypeObject THPDeviceSpecType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.DeviceSpec",                    /* tp_name */
  sizeof(THPDeviceSpec),                 /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDeviceSpec_repr,          /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPDeviceSpec_str,           /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  (richcmpfunc)THPDeviceSpec_rc,         /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPDeviceSpec_properties,              /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPDeviceSpec_pynew,                   /* tp_new */
};

void THPDeviceSpec_init(PyObject *module)
{
  if (PyType_Ready(&THPDeviceSpecType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDeviceSpecType);
  if (PyModule_AddObject(module, "DeviceSpec", (PyObject *)&THPDeviceSpecType) != 0) {
    throw python_error();
  }
}
