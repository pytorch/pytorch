#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace autograd {

struct PyAnomalyMetadata : public AnomalyMetadata {
  static constexpr const char* ANOMALY_TRACE_KEY = "traceback_";
  static constexpr const char* ANOMALY_PARENT_KEY = "parent_";

  PyAnomalyMetadata() {
    pybind11::gil_scoped_acquire gil;
    dict_ = PyDict_New();
  }
  ~PyAnomalyMetadata() override {
    // If python is already dead, leak the wrapped python objects
    if (Py_IsInitialized()) {
      pybind11::gil_scoped_acquire gil;
      Py_DECREF(dict_);
    }
  }
  void store_stack() override;
  void print_stack(const std::string& current_node_name) override;
  void assign_parent(const std::shared_ptr<Node>& parent_node) override;

  PyObject* dict() {
    return dict_;
  }

 private:
  PyObject* dict_;
};
void _print_stack(
    PyObject* trace_stack,
    const std::string& current_node_name,
    bool is_parent);

} // namespace autograd
} // namespace torch
