#include <ATen/FuncTorchTLS.h>

namespace at::functorch {

namespace {

thread_local std::unique_ptr<FuncTorchTLSBase> kFuncTorchTLS = nullptr;

}

std::unique_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS() {
  if (kFuncTorchTLS == nullptr) {
    return nullptr;
  }
  return kFuncTorchTLS->deepcopy();
}

void setFuncTorchTLS(const std::shared_ptr<const FuncTorchTLSBase>& state) {
  if (state == nullptr) {
    kFuncTorchTLS = nullptr;
    return;
  }
  kFuncTorchTLS = state->deepcopy();
}

std::unique_ptr<FuncTorchTLSBase>& functorchTLSAccessor() {
  return kFuncTorchTLS;
}


} // namespace at::functorch
