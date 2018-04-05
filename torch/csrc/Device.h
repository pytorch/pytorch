#pragma once

#include <Python.h>
#include "torch/csrc/utils/device.h"

struct THPDevice {
  PyObject_HEAD
  torch::DeviceType device_type;
  int64_t device_index;
  bool is_default;  // the default device for the type, i.e. true for 'cuda', but false for 'cuda:0'
};

extern PyTypeObject THPDeviceType;

inline bool THPDevice_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

PyObject * THPDevice_New(torch::DeviceType device_type, int64_t device_index, bool is_default);

void THPDevice_init(PyObject *module);
