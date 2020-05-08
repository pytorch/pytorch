#pragma once

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/auto_gil.h>

namespace torch { namespace autograd {

struct PyAnomalyMetadata : public AnomalyMetadata {
  static constexpr char* ANOMALY_TRACE_KEY = "traceback_";

  PyAnomalyMetadata() {
    pybind11::gil_scoped_acquire gil;
    dict_ = PyDict_New();
  }
  ~PyAnomalyMetadata() override {
    pybind11::gil_scoped_acquire gil;
    Py_DECREF(dict_);
  }
  void store_stack() override;
  void print_stack(const std::string& current_node_name) override;

  PyObject* dict() {
    return dict_;
  }

private:
  PyObject* dict_;
};

}}
