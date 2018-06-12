#pragma once

#include "torch/csrc/autograd/anomaly_mode.h"
#include "torch/csrc/python_headers.h"
#include "torch/csrc/utils/auto_gil.h"

namespace torch { namespace autograd {

#define ANOMALY_TRACE_KEY "traceback_"

struct PyAnomalyMetadata : public AnomalyMetadata {
  PyAnomalyMetadata() {
    AutoGIL gil;
    dict_ = PyDict_New();
  }
  ~PyAnomalyMetadata() {
    AutoGIL gil;
    Py_DECREF(dict_);
  }
  virtual void store_stack() override;
  virtual void print_stack() override;

  PyObject* dict() {
    return dict_;
  }

private:
  PyObject* dict_;
};

}}
