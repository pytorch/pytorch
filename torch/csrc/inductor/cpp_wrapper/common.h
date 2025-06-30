#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <utility>

#include <Python.h>
#define PYBIND11_SIMPLE_GIL_MANAGEMENT
#include <pybind11/gil.h>

// Include some often-used cpp_wrapper headers, for precompiling.
#include <c10/util/BFloat16.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/utils/pythoncapi_compat.h>
#include <torch/csrc/utils/tensor_memoryformats.h>

namespace py = pybind11; // NOLINT(misc-unused-alias-decls)

class RAIIPyObject {
 public:
  RAIIPyObject() = default;
  // steals a reference to a PyObject
  RAIIPyObject(PyObject* obj) : obj_{obj} {}
  RAIIPyObject(const RAIIPyObject& other) : obj_{other.obj_} {
    Py_XINCREF(obj_);
  }
  RAIIPyObject(RAIIPyObject&& other) noexcept {
    // refcount doesn't change, and obj_ is currently nullptr
    std::swap(obj_, other.obj_);
  }
  ~RAIIPyObject() {
    Py_XDECREF(obj_);
  }
  RAIIPyObject& operator=(const RAIIPyObject& other) {
    if (this != &other) {
      Py_XDECREF(obj_);
      obj_ = other.obj_;
      Py_XINCREF(obj_);
    }
    return *this;
  }
  RAIIPyObject& operator=(RAIIPyObject&& other) noexcept {
    // refcount to the current object decreases, but refcount to other.obj_ is
    // the same
    Py_XDECREF(obj_);
    obj_ = std::exchange(other.obj_, nullptr);
    return *this;
  }
  operator bool() const noexcept {
    return obj_;
  }
  operator PyObject*() {
    return obj_;
  }
  PyObject* get() {
    return obj_;
  }

 private:
  PyObject* obj_{nullptr};
};

#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
using namespace torch::aot_inductor;

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
