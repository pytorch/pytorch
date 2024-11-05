#include <torch/csrc/jit/backends/backend_interface.h>

namespace torch::jit {

PyTorchBackendInterface::PyTorchBackendInterface() noexcept = default;
PyTorchBackendInterface::~PyTorchBackendInterface() = default;

} // namespace torch::jit
