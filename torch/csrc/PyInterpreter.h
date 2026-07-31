#pragma once

#include <c10/core/impl/PyInterpreter.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

namespace c10 {
struct FakeTensorMode;
}

namespace torch::detail {
TORCH_PYTHON_API py::handle getTorchApiFunction(const c10::OperatorHandle& op);
}

// TODO: Move these to a proper namespace
TORCH_PYTHON_API c10::impl::PyInterpreter* getPyInterpreter();
TORCH_PYTHON_API void initializeGlobalPyInterpreter();

// Resolve the Python CppFakeTensorMode backing a C++ FakeTensorMode. The
// back-reference is weak (see fake_mode_pyobj_), so this returns None when the
// mode is unset or the Python object has been collected.
TORCH_PYTHON_API py::object getFakeModePyObj(const c10::FakeTensorMode* mode);
