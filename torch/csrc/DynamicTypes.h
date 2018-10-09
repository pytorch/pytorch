#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include "torch/csrc/python_headers.h"

#include <ATen/Device.h>
#include <ATen/core/ScalarType.h>
#include <ATen/core/Backend.h>

#include <memory>
#include <string>

struct THPDtype;
struct THPLayout;

namespace at {
struct Storage;
struct Type;
} // namespace at

namespace torch {
// Register a PyTypeObject* with the given attributes
void registerStoragePyTypeObject(
    PyTypeObject *pytype, const std::string& name,
    bool is_cuda, bool is_sparse);

void registerDtypeObject(THPDtype *dtype, at::ScalarType scalarType);
void registerLayoutObject(THPLayout *layout, at::Backend backend);

PyObject* createPyObject(const at::Storage& storage);
at::Storage createStorage(PyObject* obj);
bool isStorage(PyObject* obj);

THPDtype* getDtype(at::ScalarType scalarType);
THPLayout* getLayout(at::Backend backend);
at::Type& getVariableType(at::ScalarType scalarType, const THPLayout& layout, const at::Device& device);
at::Device::Type getDeviceType(const at::Type& type);
}  // namespace torch
