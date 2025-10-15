#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::detail {
TORCH_PYTHON_API py::handle getTorchApiFunction(const c10::OperatorHandle& op);
}

// TODO: Move these to a proper namespace
TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
TORCH_PYTHON_API void initializeGlobalPyInterpreter();
