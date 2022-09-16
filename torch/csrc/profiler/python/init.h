#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>

#include <torch/csrc/profiler/collection.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_numbers.h>

namespace pybind11 {
namespace detail {
using torch::profiler::impl::StorageImplData;
using torch::profiler::impl::TensorImplAddress;

// Strong typedefs don't make much sense in Python since everything is duck
// typed. So instead we simply cast them to ints, return them, and let the
// caller handle correctness.
template <typename T>
struct strong_pointer_type_caster {
  template <typename T_>
  static handle cast(T_&& src, return_value_policy policy, handle parent) {
    const auto* ptr = reinterpret_cast<const void*>(src.value_of());
    return ptr ? handle(THPUtils_packUInt64(reinterpret_cast<intptr_t>(ptr)))
               : none();
  }

  bool load(handle src, bool convert) {
    return false;
  }

  PYBIND11_TYPE_CASTER(T, _("strong_pointer"));
};

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
