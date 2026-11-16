#pragma once

#include <c10/core/FakeTensorMode.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace torch::fake_tensor {

TORCH_PYTHON_API extern PyTypeObject PyCppFakeTensorModeType;

// Accepts only the exact CppFakeTensorMode. Caller must hold the GIL.
TORCH_PYTHON_API std::shared_ptr<c10::FakeTensorMode> getCppFakeTensorMode(
    py::handle py_mode);

// Reuses the live Python wrapper or rebuilds it around mode; null gives None.
// Caller must hold the GIL.
TORCH_PYTHON_API py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode);

} // namespace torch::fake_tensor
