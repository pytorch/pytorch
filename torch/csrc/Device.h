#pragma once

#include <torch/csrc/python_headers.h>

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

namespace torch { namespace python {

/**
 * Utility for parsing Python device argument.
 */
at::Device parseDevice(py::handle device);

void initDeviceBindings(PyObject* module);

}} // namespace torch::python

// Legacy functions are still kept to ease transition to pybind11 bindings
// FIXME Remove use of these functions and get rid of them
inline bool THPDevice_Check(PyObject *obj) {
  return py::isinstance<at::Device>(obj);
}

inline PyObject* THPDevice_New(const at::Device& device) {
  return py::cast(device).release().ptr();
}
