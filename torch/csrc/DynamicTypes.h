#pragma once

// Provides conversions between Python tensor objects and thpp::Tensors.

#include <Python.h>
#include <memory>
#include <unordered_map>
#include <ATen/ATen.h>

namespace torch {

// Register a PyTypeObject* with the given attributes
void registerPyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

PyObject* createPyObject(at::Tensor& tensor);
PyTypeObject* getPyTypeObject(const at::Tensor& tensor);
//rename to createPyObject when THPP is removed
at::Tensor createTensor(PyObject *data);

}  // namespace torch
