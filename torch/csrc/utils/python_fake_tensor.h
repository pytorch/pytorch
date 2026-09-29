#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace torch {

// get the python CppFakeTensorMode object for mode, minting a fresh wrapper
// around the same C++ mode if the previous one has been collected. Returns None
// only when mode is null.
// Caller must hold the GIL.
TORCH_PYTHON_API py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode);

} // namespace torch
