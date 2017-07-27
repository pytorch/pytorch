#pragma once

// Provides conversions between Python tensor objects and thpp::Tensors.

#include <Python.h>
#include <memory>
#include <THPP/THPP.h>
#include <ATen/ATen.h>

namespace torch {

// Register a PyTypeObject* with the given attributes
void registerPyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

// Gets the PyTypeObject* corresponding to the Tensor
PyTypeObject* getPyTypeObject(const thpp::Tensor& tensor);

// Creates a Tensor from a Python tensor object
std::unique_ptr<thpp::Tensor> createTensor(PyObject *data);

// Creates Python tensor object from a Tensor
PyObject* createPyObject(const thpp::Tensor& tensor);

PyObject* createPyObject(at::Tensor& tensor);
PyTypeObject* getPyTypeObject(const at::Tensor& tensor);
//rename to createPyObject when THPP is removed
at::Tensor createTensorAT(PyObject *data);

}  // namespace torch
