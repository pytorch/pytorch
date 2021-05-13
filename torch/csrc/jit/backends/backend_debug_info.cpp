#include <c10/macros/Macros.h>

#ifndef C10_MOBILE
#include <torch/csrc/jit/backends/backend_debug_info.h>

namespace torch {
namespace jit {
namespace backend {
namespace {
static auto cls =
    torch::class_<PyTorchBackendDebugInfo>("backend", "BackendDebugInfo")
        .def(torch::init<>());

} // namespace
} // namespace backend
} // namespace jit
} // namespace torch

#else

namespace torch {
namespace jit {
namespace backend {
namespace {

class PyTorchBackendDebugInfoDummy : public torch::CustomClassHolder {
 public:
  PyTorchBackendDebugInfoDummy() = default;
};
static auto cls =
    torch::class_<PyTorchBackendDebugInfoDummy>("backend", "BackendDebugInfo")
        .def(torch::init<>());

} // namespace
} // namespace backend
} // namespace jit
} // namespace torch
#endif
