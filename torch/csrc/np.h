#pragma once


#include <torch/csrc/python_headers.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/tensor_new.h>

struct THNPArray : public THPVariable {
};

TORCH_API extern PyTypeObject THNPArrayBaseType;

TORCH_API extern PyObject *THNPArrayClass;

inline bool THNPArray_Check(PyObject *obj) {
  return Py_TYPE(obj) == (PyTypeObject *)THNPArrayClass;
}
#if PY_MAJOR_VERSION == 2
#define NP_H_IntType PyInt_Type
#else
#define NP_H_IntType PyLong_Type
#endif
inline bool THNP_checkDtype(PyObject *obj) {
  if (PyType_Check(obj)) {
      PyTypeObject *to = (PyTypeObject*)obj;
      return to == &NP_H_IntType
        || to == &PyFloat_Type;
  }
  return false;
}
PyObject* THNPArray_NewWithTensor(PyTypeObject* type, torch::autograd::Variable v);
