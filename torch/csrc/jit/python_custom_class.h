#pragma once

#include <torch/csrc/jit/pybind_utils.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {

void initPythonCustomClassBindings(PyObject* module);

}
} // namespace torch
