#include <torch/csrc/jit/backends/backend_interface.h>

namespace torch {
namespace jit {

PyTorchBackendInterface::PyTorchBackendInterface() noexcept = default;
PyTorchBackendInterface::~PyTorchBackendInterface() = default;

} // namespace jit
} // namespace torch
