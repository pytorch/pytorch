#pragma once

#include <array>
#include <filesystem>
#include <optional>

#include <Python.h>
#define PYBIND11_SIMPLE_GIL_MANAGEMENT
#include <pybind11/gil.h>
namespace py = pybind11;

class RAIIPyObject {
 public:
  RAIIPyObject() : obj_(nullptr) {}
  RAIIPyObject(PyObject* obj) : obj_(obj) {}
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
  operator PyObject*() {
    return obj_;
  }
  PyObject* get() {
    return obj_;
  }

 private:
  PyObject* obj_;
};

#include <torch/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch/csrc/inductor/aoti_runtime/utils.h>
using namespace torch::aot_inductor;

#include <c10/util/generic_math.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
using half = at::Half;
using bfloat16 = at::BFloat16;

// Round up to the nearest multiple of 64
[[maybe_unused]] inline int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}
