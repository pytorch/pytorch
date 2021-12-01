#pragma once

#include <memory>
#include <c10/macros/Macros.h>

namespace at { namespace functorch {

// NOTE [functorch TLS in pytorch/pytorch]
//
// functorch lives out-of-tree. However, it has some TLS that needs to be
// propagated. The solution for that is we store a pointer to the TLS
// inside pytorch/pytorch and extend FuncTorchTLSBase inside functorch to
// include whatever functorch needs.
//
// A previous solution used ThreadLocalDebugInfo. However, all
// PyTorch-managed threads (e.g. spawned by Autograd) all receive a
// shared_ptr that points to the same ThreadLocalDebugInfo. This leads to
// race conditions if the multiple threads start modifying the TLS
// stored within ThreadLocalDebugInfo.
struct TORCH_API FuncTorchTLSBase {
  virtual ~FuncTorchTLSBase() = default;
  virtual std::shared_ptr<FuncTorchTLSBase> deepcopy() = 0;
};

TORCH_API std::shared_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS();
TORCH_API void setFuncTorchTLS(const std::shared_ptr<FuncTorchTLSBase>& state);

TORCH_API std::shared_ptr<FuncTorchTLSBase>& functorchTLSAccessor();

}}
