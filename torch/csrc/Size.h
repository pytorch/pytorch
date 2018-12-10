#pragma once

#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/variable.h>
#include <cstdint>

extern PyTypeObject THPSizeType;

#define THPSize_Check(obj) (Py_TYPE(obj) == &THPSizeType)

PyObject * THPSize_New(const torch::autograd::Variable& t);
PyObject * THPSize_NewFromSizes(int dim, const int64_t *sizes);

#ifdef _THP_CORE
void THPSize_init(PyObject *module);
#endif
