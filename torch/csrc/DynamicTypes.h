#pragma once

// Provides conversions between Python tensor objects and at::Tensor.

#include <torch/csrc/python_headers.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Backend.h>
#include <c10/core/Layout.h>

#include <memory>
#include <string>

struct THPLayout;

namespace c10 {
struct Storage;
}

namespace torch {
// Register a PyTypeObject* with the given attributes
void registerStoragePyTypeObject(
    PyTypeObject *pytype, at::Backend backend, at::ScalarType scalarType);

void registerLayoutObject(THPLayout *thp_layout, at::Layout layout);

PyObject* createPyObject(
    const at::Storage& storage,
    const caffe2::TypeMeta data_type);
at::Storage createStorage(PyObject* obj);
bool isStorage(PyObject* obj);

THPLayout* getTHPLayout(at::Layout layout);

}  // namespace torch
