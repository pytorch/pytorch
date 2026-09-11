#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::autograd {

struct PyAnomalyMetadata : public AnomalyMetadata {
  static constexpr const char* ANOMALY_TRACE_KEY = "traceback_";
  static constexpr const char* ANOMALY_PARENT_KEY = "parent_";

  PyAnomalyMetadata() {
    pybind11::gil_scoped_acquire gil;
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    dict_ = PyDict_New();
  }
  ~PyAnomalyMetadata() override {
    // Leak the wrapped python object if the GIL can't be acquired (e.g.
    // python is already dead).
    torch::detail::SafeGilScopedAcquire gil;
    if (!gil) {
      return;
    }
    Py_DECREF(dict_);
  }
  void store_stack() override;
  void print_stack(const std::string& current_node_name) override;
  void assign_parent(const c10::intrusive_ptr<Node>& parent_node) override;

  PyObject* dict() {
    return dict_;
  }

 private:
  PyObject* dict_{nullptr};
};
void _print_stack(
    PyObject* trace_stack,
    const std::string& current_node_name,
    bool is_parent);

} // namespace torch::autograd
