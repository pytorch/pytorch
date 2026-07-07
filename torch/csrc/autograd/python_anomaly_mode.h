#pragma once

#include <torch/csrc/python_headers.h>

#include <c10/core/SafePyObject.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::autograd {

struct PyAnomalyMetadata : public AnomalyMetadata {
  static constexpr const char* ANOMALY_TRACE_KEY = "traceback_";
  static constexpr const char* ANOMALY_PARENT_KEY = "parent_";

  // dict_ is a SafePyObject so destruction routes through
  // PyInterpreter::decref instead of a hand-written destructor here.
  PyAnomalyMetadata()
      : dict_(
            [] {
              pybind11::gil_scoped_acquire gil;
              return PyDict_New();
            }(),
            getPyInterpreter()) {}
  ~PyAnomalyMetadata() override = default;

  void store_stack() override;
  void print_stack(const std::string& current_node_name) override;
  void assign_parent(const c10::intrusive_ptr<Node>& parent_node) override;

  PyObject* dict() {
    return dict_.ptr(getPyInterpreter());
  }

 private:
  c10::SafePyObject dict_;
};
void _print_stack(
    PyObject* trace_stack,
    const std::string& current_node_name,
    bool is_parent);

} // namespace torch::autograd
