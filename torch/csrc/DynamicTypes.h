#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <Python.h>
#include <memory>
#include <unordered_map>
#include <ATen/ATen.h>

namespace torch {

// Register a PyTypeObject* with the given attributes
void registerPyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

PyObject* createPyObject(const at::Tensor& tensor);
PyTypeObject* getPyTypeObject(const at::Tensor& tensor);
//rename to createPyObject when THPP is removed
// Creates a at::Tensor from a PyObject.  Does NOT steal the PyObject reference.
at::Tensor createTensor(PyObject *data);

}  // namespace torch
