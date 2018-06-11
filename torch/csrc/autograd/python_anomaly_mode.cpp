#include "torch/csrc/autograd/python_anomaly_mode.h"
#include "torch/csrc/python_headers.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"

#include <iostream>

namespace torch { namespace autograd {

void PyAnomalyMetadata::store_stack() {
  AutoGIL gil;
  THPObjectPtr mod(PyImport_ImportModule("traceback"));
  if (!mod) {
    throw python_error();
  }

  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));

  if (PyDict_SetItemString(dict(), ANOMALY_TRACE_KEY, list.get())) {
    throw python_error();
  }
}

void PyAnomalyMetadata::print_stack() {
  AutoGIL gil;
  if (!PyDict_Check(dict())) {
    throw std::runtime_error("Anomaly metadata is not a python dictionary.");
  }

  THPObjectPtr stack(PyDict_GetItemString(dict(), ANOMALY_TRACE_KEY));
  if (!stack) {
    std::cout << "No forward pass information available." << std::endl;
    std::cout << "Enable detect anomaly during forward pass for more informations." << std::endl;
    return;
  }

  THPObjectPtr empty_string(PyUnicode_FromString(""));
  THPObjectPtr msg(PyUnicode_Join(empty_string, stack.get()));

  std::cout << "Traceback of forward call that caused the error:" << std::endl;
  std::cout << THPUtils_unpackString(msg.get()) << std::endl;
}

}}
