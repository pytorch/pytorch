#pragma once

#include <Python.h>

#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/python/pybind.h>

namespace pybind11 {
namespace detail {
using torch::profiler::impl::StorageImplData;
using torch::profiler::impl::TensorImplAddress;

template <>
struct type_caster<StorageImplData>
    : public strong_pointer_type_caster<StorageImplData> {};

template <>
struct type_caster<TensorImplAddress>
    : public strong_pointer_type_caster<TensorImplAddress> {};
} // namespace detail
} // namespace pybind11

namespace torch {
namespace profiler {

void initPythonBindings(PyObject* module);

} // namespace profiler
} // namespace torch
