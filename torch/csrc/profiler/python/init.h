#pragma once

#include <Python.h>

#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/profiler/python/pybind.h>

namespace pybind11 {
namespace detail {
using torch::profiler::impl::TensorID;

#define STRONG_POINTER_TYPE_CASTER(T) \
  template <>                         \
  struct type_caster<T> : public strong_pointer_type_caster<T> {};

STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::StorageImplData);
STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::AllocationID);
STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::TensorImplAddress);
STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::PyModuleSelf);
STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::PyModuleCls);
STRONG_POINTER_TYPE_CASTER(torch::profiler::impl::PyOptimizerSelf);
#undef STRONG_POINTER_TYPE_CASTER

template <>
struct type_caster<TensorID> : public strong_uint_type_caster<TensorID> {};
} // namespace detail
} // namespace pybind11

namespace torch {
namespace profiler {

void initPythonBindings(PyObject* module);

} // namespace profiler
} // namespace torch
