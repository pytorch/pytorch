#pragma once

#include <Python.h>
#include "torch/csrc/utils/device.h"

struct THPDevice {
  PyObject_HEAD
  torch::Device device;
};

extern PyTypeObject THPDeviceType;

inline bool THPDevice_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

PyObject * THPDevice_New(const torch::Device& device);

void THPDevice_init(PyObject *module);
