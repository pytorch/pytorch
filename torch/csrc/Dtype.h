#pragma once

#include <torch/csrc/python_headers.h>
#include <c10/util/string_view.h>
#include <ATen/ATen.h>

namespace torch {

struct PyDtype {
  bool defined;
  at::ScalarType scalar_type;
  c10::string_view primary_name;
  c10::string_view legacy_name;
};

const PyDtype& getPyDtype(at::ScalarType scalar_type);

void initDtypeBindings(PyObject* module);

} // namespace torch

// Legacy functions are still kept to ease transition to pybind11 bindings
// FIXME Remove use of these functions and get rid of them
bool THPDtype_Check(PyObject* obj);

inline bool THPPythonScalarType_Check(PyObject *obj) {
  return obj == (PyObject*)(&PyFloat_Type) ||
    obj == (PyObject*)(&PyBool_Type) ||
    obj == (PyObject*)(&PyLong_Type);
}
