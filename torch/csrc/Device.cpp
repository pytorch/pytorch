#include "Device.h"

#include <cstring>
#include <structmember.h>
#include <sstream>
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/python_strings.h"

PyObject *THPDevice_New(const torch::Device& device)
{
  auto type = (PyTypeObject*)&THPDeviceType;
  auto self = THPObjectPtr{type->tp_alloc(type, 0)};
  if (!self) throw python_error();
  auto self_ = reinterpret_cast<THPDevice*>(self.get());
  self_->device = device;
  return self.release();
}

static const char* cuda_str = "cuda";
static const char* cpu_str = "cpu";

static inline const char* deviceTypeString(torch::DeviceType device_type) {
  switch (device_type) {
    case torch::DeviceType::CUDA:
      return cuda_str;
    case torch::DeviceType::CPU:
      return cpu_str;
    default:
      throw std::runtime_error("unexpected device type");
  }
}

PyObject *THPDevice_repr(THPDevice *self)
{
  std::ostringstream oss;
  oss << "device(device_type=\'" << deviceTypeString(self->device.type) << "\'";
  if (!self->device.is_default) {
    oss << ", device_index=" << self->device.index;
  }
  oss << ")";
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDevice_str(THPDevice*self)
{
  std::ostringstream oss;
  if (!self->device.is_default) {
    oss << deviceTypeString(self->device.type) << ":" << self->device.index;
  } else {
    oss << deviceTypeString(self->device.type);
  }
  return THPUtils_packString(oss.str().c_str());
}

PyObject *THPDevice_pynew(PyTypeObject *type, PyObject *args, PyObject *kwargs)
{
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({
    "Device(Device device)",
    "Device(String device_type, int64_t? device_index=-1)"
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.idx == 0) {
    auto device = r.device(0);
    return THPDevice_New(device);
  } else if (r.idx == 1) {
    auto as_device = r.device(0);  // this works, because device can take strings
    auto device_type = r.string(0);
    if (!as_device.is_default) {
      throw std::runtime_error("device_type (string) must not include an index because index "
                                "was passed explicitly: " + device_type);
    }

    auto is_default = r.isNone(1);
    auto device_index = r.toInt64WithDefault(1, -1);
    // make sure this is constructible
    auto device = torch::Device(as_device.type, device_index, is_default);
    return THPDevice_New(device);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDevice_type(THPDevice *self)
{
  HANDLE_TH_ERRORS
  return THPUtils_packString(deviceTypeString(self->device.type));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject *THPDevice_index(THPDevice *self)
{
  HANDLE_TH_ERRORS
  if (self->device.is_default) {
    Py_RETURN_NONE;
  } else {
    return THPUtils_packInt64(self->device.index);
  }
  END_HANDLE_TH_ERRORS
}

PyObject *THPDevice_rc(PyObject *a, PyObject *b, int op) {
  HANDLE_TH_ERRORS
  if (!THPDevice_Check(a) || !THPDevice_Check(b)) {
    // Py_RETURN_NOTIMPLEMENTED not in python 2.
    Py_INCREF(Py_NotImplemented);
    return Py_NotImplemented;
  }
  THPDevice *da = reinterpret_cast<THPDevice*>(a);
  THPDevice *db = reinterpret_cast<THPDevice*>(b);

  switch(op) {
    case Py_EQ:
      if (da->device == db->device) {
        Py_RETURN_TRUE;
      } else {
        Py_RETURN_FALSE;
      }
    case Py_NE:
      if (da->device == db->device) {
        Py_RETURN_FALSE;
      } else {
        Py_RETURN_TRUE;
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

static struct PyGetSetDef THPDevice_properties[] = {
  {"type",       (getter)THPDevice_type, nullptr, nullptr, nullptr},
  {"index",      (getter)THPDevice_index, nullptr, nullptr, nullptr},
  {nullptr}
};

PyTypeObject THPDeviceType = {
  PyVarObject_HEAD_INIT(nullptr, 0)
  "torch.Device",                        /* tp_name */
  sizeof(THPDevice),                     /* tp_basicsize */
  0,                                     /* tp_itemsize */
  0,                                     /* tp_dealloc */
  0,                                     /* tp_print */
  0,                                     /* tp_getattr */
  0,                                     /* tp_setattr */
  0,                                     /* tp_reserved */
  (reprfunc)THPDevice_repr,              /* tp_repr */
  0,                                     /* tp_as_number */
  0,                                     /* tp_as_sequence */
  0,                                     /* tp_as_mapping */
  0,                                     /* tp_hash  */
  0,                                     /* tp_call */
  (reprfunc)THPDevice_str,               /* tp_str */
  0,                                     /* tp_getattro */
  0,                                     /* tp_setattro */
  0,                                     /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                    /* tp_flags */
  nullptr,                               /* tp_doc */
  0,                                     /* tp_traverse */
  0,                                     /* tp_clear */
  (richcmpfunc)THPDevice_rc,             /* tp_richcompare */
  0,                                     /* tp_weaklistoffset */
  0,                                     /* tp_iter */
  0,                                     /* tp_iternext */
  0,                                     /* tp_methods */
  0,                                     /* tp_members */
  THPDevice_properties,                  /* tp_getset */
  0,                                     /* tp_base */
  0,                                     /* tp_dict */
  0,                                     /* tp_descr_get */
  0,                                     /* tp_descr_set */
  0,                                     /* tp_dictoffset */
  0,                                     /* tp_init */
  0,                                     /* tp_alloc */
  THPDevice_pynew,                       /* tp_new */
};

void THPDevice_init(PyObject *module)
{
  if (PyType_Ready(&THPDeviceType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THPDeviceType);
  if (PyModule_AddObject(module, "device", (PyObject *)&THPDeviceType) != 0) {
    throw python_error();
  }
}
