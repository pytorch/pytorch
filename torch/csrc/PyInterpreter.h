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

// get the python CppFakeTensorMode object for mode, minting a fresh wrapper
// around the same C++ mode if the previous one has been collected. Returns None
// only when mode is null.
TORCH_PYTHON_API py::object getFakeModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode);
