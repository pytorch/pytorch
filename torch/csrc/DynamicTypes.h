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

// Register a PyTypeObject* with the given attributes
void registerStoragePyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

PyObject* createPyObject(const at::Tensor& tensor);
PyObject* createPyObject(const at::Storage& storage);
PyTypeObject* getPyTypeObject(const at::Tensor& tensor);
at::Type& getATenType(PyTypeObject* type);
//rename to createPyObject when THPP is removed
// Creates a at::Tensor from a PyObject.  Does NOT steal the PyObject reference.
at::Tensor createTensor(PyObject* data);
std::unique_ptr<at::Storage> createStorage(PyObject* obj);

bool isStorage(PyObject* obj);

}  // namespace torch
