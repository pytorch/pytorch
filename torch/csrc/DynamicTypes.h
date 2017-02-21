#pragma once

// Provides conversions between Python tensor objects and thpp::Tensors.

#include <memory>
#include <Python.h>
#include <THPP/THPP.h>

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

}  // namespace torch
