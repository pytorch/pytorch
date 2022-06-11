#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <torch/csrc/python_headers.h>

#include <ATen/Device.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>

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

PyObject* createPyObject(const at::Storage& storage);
at::Storage createStorage(PyObject* obj);
at::Storage createStorageGetType(
    PyObject* obj,
    at::ScalarType& scalar_type,
    bool& is_typed_storage);
bool isStorage(PyObject* obj);

THPDtype* getTHPDtype(at::ScalarType scalarType);
THPLayout* getTHPLayout(at::Layout layout);
} // namespace torch
