#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace torch::fake_tensor {

// The handle of py_mode's C++ mode, the one its fakes and the fake mode slot
// share while any of them uses it; see Note [C++ FakeTensorMode handle] in
// FakeTensorMode.cpp. Raises TypeError unless py_mode is a CppFakeTensorMode
// (not a subclass, nor a bare _CppFakeTensorMode).
// Caller must hold the GIL.
TORCH_PYTHON_API std::shared_ptr<c10::FakeTensorModeHandle>
getCppFakeTensorModeHandle(py::handle py_mode);

// The Python CppFakeTensorMode object of handle's mode, or None if handle is
// null. Raises for a mode made in C++, which has none.
// Caller must hold the GIL.
TORCH_PYTHON_API py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorModeHandle>& handle);

} // namespace torch::fake_tensor
