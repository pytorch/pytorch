#pragma once

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/python_headers.h>
#include <cstdint>

extern PyTypeObject THPSizeType;

#define THPSize_Check(obj) (Py_TYPE(obj) == &THPSizeType)

PyObject* THPSize_New(const torch::autograd::Variable& t);
PyObject* THPSize_NewFromSizes(int64_t dim, const int64_t* sizes);
PyObject* THPSize_NewFromSymSizes(const at::Tensor& t);

void THPSize_init(PyObject* module);
