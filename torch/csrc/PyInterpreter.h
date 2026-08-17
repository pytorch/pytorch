#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

namespace c10 {
struct FakeTensorMode; // c++ faketensormode
}

namespace torch::detail {
TORCH_PYTHON_API py::handle getTorchApiFunction(const c10::OperatorHandle& op);
}

// TODO: Move these to a proper namespace
TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
TORCH_PYTHON_API void initializeGlobalPyInterpreter();

// get the python CppFakeTensorMode object, returns None when mode is unset or
// python object no longer exists
TORCH_PYTHON_API py::object getFakeModePyObj(const c10::FakeTensorMode* mode);
