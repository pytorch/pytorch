#pragma once

#include "torch/csrc/python_headers.h"

#include <ATen/Device.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct THPDevice {
  PyObject_HEAD
  at::Device device;
};

extern PyTypeObject THPDeviceType;

inline bool THPDevice_Check(PyObject *obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

PyObject * THPDevice_New(const at::Device& device);

void THPDevice_init(PyObject *module);
