#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch_lazy_tensors {
namespace compiler {

torch::lazy::BackendImplInterface* GetTSBackendImpl();

void InitTorchScriptBackend();

}  // namespace compiler
}  // namespace torch_lazy_tensors
