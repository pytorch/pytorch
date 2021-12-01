#include <ATen/FuncTorchTLS.h>

namespace at { namespace functorch {

thread_local std::shared_ptr<FuncTorchTLSBase> kFuncTorchTLS = nullptr;

std::shared_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS() {
  if (kFuncTorchTLS == nullptr) {
    return nullptr;
  }
  return kFuncTorchTLS->deepcopy();
}

void setFuncTorchTLS(const std::shared_ptr<FuncTorchTLSBase>& state) {
  if (state == nullptr) {
    kFuncTorchTLS = nullptr;
    return;
  }
  kFuncTorchTLS = state->deepcopy();
}

std::shared_ptr<FuncTorchTLSBase>& functorchTLSAccessor() {
  return kFuncTorchTLS;
}


}}
