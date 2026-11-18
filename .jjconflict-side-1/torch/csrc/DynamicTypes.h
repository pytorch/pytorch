#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <torch/csrc/python_headers.h>

#include <ATen/Device.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/csrc/Export.h>

#include <memory>
#include <string>

struct THPDtype;
struct THPLayout;

namespace c10 {
struct Storage;
}

namespace torch {
void registerDtypeObject(THPDtype* dtype, at::ScalarType scalarType);
void registerLayoutObject(THPLayout* thp_layout, at::Layout layout);

TORCH_PYTHON_API PyObject* createPyObject(const at::Storage& storage);
TORCH_PYTHON_API at::Storage createStorage(PyObject* obj);
TORCH_PYTHON_API std::tuple<at::Storage, at::ScalarType, bool>
createStorageGetType(PyObject* obj);
TORCH_PYTHON_API bool isStorage(PyObject* obj);

// Both methods below return a borrowed reference!
TORCH_PYTHON_API THPDtype* getTHPDtype(at::ScalarType scalarType);
TORCH_PYTHON_API THPLayout* getTHPLayout(at::Layout layout);
} // namespace torch
