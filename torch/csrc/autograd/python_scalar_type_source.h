#pragma once

#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <ATen/ATen.h>

inline c10::optional<at::ScalarTypeSource> THPUtils_scalarTypeSource(PyObject *obj) {
  if (THPVariable_Check(obj)) {
    return at::ScalarTypeSource(reinterpret_cast<THPVariable*>(obj)->cdata);
  } else if (PyBool_Check(obj)) {
    return at::ScalarTypeSource(obj == Py_True);
  } else if (THPUtils_checkLong(obj)) {
    return (at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(obj))));
  } else if (THPUtils_checkDouble(obj)) {
    return at::ScalarTypeSource(at::Scalar(THPUtils_unpackDouble(obj)));
  } else if (PyComplex_Check(obj)) {
    return at::ScalarTypeSource(at::Scalar(THPUtils_unpackComplexDouble(obj)));
  } else if (THPDtype_Check(obj)) {
    return at::ScalarTypeSource(reinterpret_cast<THPDtype*>(obj)->scalar_type);
  } else {
    return c10::nullopt;
  }
}
