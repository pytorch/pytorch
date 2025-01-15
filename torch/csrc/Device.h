#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

#include <ATen/Device.h>

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct TORCH_API THPDevice {
  PyObject_HEAD
  at::Device device;
};

TORCH_API extern PyTypeObject THPDeviceType;

inline bool THPDevice_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPDeviceType;
}

TORCH_API PyObject* THPDevice_New(const at::Device& device);

TORCH_API void THPDevice_init(PyObject* module);
