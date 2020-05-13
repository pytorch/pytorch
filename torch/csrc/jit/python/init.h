#pragma once

#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

void initJITBindings(PyObject* module);

}
} // namespace torch
