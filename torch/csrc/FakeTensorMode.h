#pragma once

#include <c10/core/TensorImpl.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/utils/pybind.h>

#include <memory>

namespace torch::fake_tensor {

// get the python CppFakeTensorMode object for mode, minting a fresh one around
// the same C++ mode if the previous one has been collected; see
// Note [C++ FakeTensorMode Python object lifetime] in FakeTensorMode.cpp.
// Returns None only when mode is null.
// Caller must hold the GIL.
TORCH_PYTHON_API py::object getCppFakeTensorModePyObj(
    const std::shared_ptr<c10::FakeTensorMode>& mode);

// The mode in a capsule that getCppFakeTensorModePyObj passes to
// _CppFakeTensorMode's constructor to rebuild a mode's collected Python object;
// see Note [C++ FakeTensorMode Python object lifetime].
std::shared_ptr<c10::FakeTensorMode> cpp_fake_mode_from_capsule(
    const py::capsule& capsule);

} // namespace torch::fake_tensor
