#pragma once

#include <Python.h>
#include "torch/csrc/utils/device.h"

struct THPDeviceSpec {
  PyObject_HEAD
  torch::DeviceType device_type;
  int64_t device_index;
  bool is_default;
};

extern PyTypeObject THPDeviceSpecType;

inline bool THPDeviceSpec_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceSpecType;
}

PyObject * THPDeviceSpec_New(torch::DeviceType device_type, int64_t device_index, bool is_default);

void THPDeviceSpec_init(PyObject *module);
