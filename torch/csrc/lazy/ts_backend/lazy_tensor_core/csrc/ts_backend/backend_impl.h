#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

namespace torch_lazy_tensors {
namespace compiler {

BackendImplInterface* GetTSBackendImpl();

void InitTorchScriptBackend();

}  // namespace compiler
}  // namespace torch_lazy_tensors
