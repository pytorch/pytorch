#pragma once
#include <pybind11/pybind11.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace lazy {

TORCH_PYTHON_API void initLazyBindings(PyObject* module);

} // namespace lazy
} // namespace torch
