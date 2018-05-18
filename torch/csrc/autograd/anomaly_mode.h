#pragma once

#include "torch/csrc/utils/python_stub.h"

namespace torch { namespace autograd {

struct AnomalyMode {
  static bool is_enabled() {
    return _enabled;
  }
  static void set_enabled(bool enabled) {
    _enabled = enabled;
  }

  static void store_stack(PyObject* metadata);
  static void print_stack(PyObject* metadata);

private:
  static bool _enabled;
  static PyObject* stacktrace_key;
};

}}
