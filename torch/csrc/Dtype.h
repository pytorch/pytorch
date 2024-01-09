#pragma once

#include <c10/core/ScalarType.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/python_headers.h>

constexpr int DTYPE_NAME_LEN = 64;

struct TORCH_API THPDtype {
  PyObject_HEAD at::ScalarType scalar_type;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  char name[DTYPE_NAME_LEN + 1];
};

TORCH_API extern PyTypeObject THPDtypeType;

inline bool THPDtype_Check(PyObject* obj) {
  return Py_TYPE(obj) == &THPDtypeType;
}

inline bool THPPythonScalarType_Check(PyObject* obj) {
  return obj == (PyObject*)(&PyFloat_Type) ||
      obj == (PyObject*)(&PyBool_Type) || obj == (PyObject*)(&PyLong_Type);
}

TORCH_API PyObject* THPDtype_New(
    at::ScalarType scalar_type,
    const std::string& name);

void THPDtype_init(PyObject* module);

// this is a private function that instantiate a custom dtype that is backed up
// by some existing dtypes, e.g. int4 backed up by bits8, this will be used in
// Tensor subclass for defining a Tensor with a new dtype, there is no gurantee
// that this dtype will work with any features, all features will be supported
// on demand
PyObject* THPDtype_pyNewCustomDtype(
    PyObject* type,
    PyObject* args,
    PyObject* kwargs);
