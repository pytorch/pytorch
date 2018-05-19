#include "torch/csrc/python_headers.h"
#include "torch/csrc/autograd/anomaly_mode.h"
#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/python_strings.h"
#include "torch/csrc/utils/object_ptr.h"
#include "torch/csrc/Exceptions.h"

#include <stdexcept>
#include <iostream>

namespace torch { namespace autograd {

bool AnomalyMode::_enabled = 0;
PyObject* AnomalyMode::stacktrace_key = THPUtils_packString("stacktrace_");

void AnomalyMode::store_stack(PyObject* metadata) {
  AutoGIL gil;
  THPObjectPtr mod(PyImport_ImportModule("traceback"));
  if (!mod) {
    throw python_error();
  }

  THPObjectPtr list(PyObject_CallMethod(mod.get(), "format_stack", ""));

  if (PyDict_SetItem(metadata, AnomalyMode::stacktrace_key, list.get())) {
    throw python_error();
  }
}

void AnomalyMode::print_stack(PyObject* metadata) {
  AutoGIL gil;
  if (!PyDict_Check(metadata)) {
    throw std::runtime_error("Function metadata is not a python dictionary.");
  }

  if (!PyDict_Contains(metadata, AnomalyMode::stacktrace_key)) {
    std::cout << "No forward pass information available." << std::endl;
    std::cout << "Enable detect anomaly during forward pass for more informations." << std::endl;
    return;
  }

  THPObjectPtr empty_string(PyUnicode_FromString(""));
  THPObjectPtr msg(PyUnicode_Join(empty_string, PyDict_GetItem(metadata, AnomalyMode::stacktrace_key)));

  std::cout << "Traceback of forward call that caused the error:" << std::endl;
  std::cout << THPUtils_unpackString(msg) << std::endl;
}

}}
