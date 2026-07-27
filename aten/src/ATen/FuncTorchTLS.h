#pragma once

#include <c10/macros/Macros.h>
#include <memory>

namespace at::functorch {

// NOTE [functorch TLS in pytorch/pytorch]
//
// functorch lives out-of-tree. However, it has some TLS that needs to be
// propagated. The solution for that is we store a pointer to the TLS
// inside pytorch/pytorch and extend FuncTorchTLSBase inside functorch to
// include whatever functorch needs.
//
// We need to store a pointer due to the indirection:
// inside functorch, we will create a subclass of FunctorchTLSBase called
// FuncTorchTLSImpl that actually contains metadata, like the DynamicLayerStack.
// FuncTorchTLSBase doesn't have any metadata because it hasn't been defined
// yet.
//
// Here in pytorch/pytorch, we will pass around FuncTorchTLSBase*, but inside
// functorch, we will assign a FuncTorchTLSImpl* to the FunctorchTLSBase*.
// We can't directly pass around FunctorchTLSBase (without a pointer) because
// FuncTorchTLSImpl does not fit inside a FuncTorchTLSBase by virtue of having
// more elements.
struct TORCH_API FuncTorchTLSBase {
  virtual ~FuncTorchTLSBase() = default;
  virtual std::unique_ptr<FuncTorchTLSBase> deepcopy() const = 0;

  virtual int64_t checkSupportsSingleLevelAutogradFunction() const = 0;
  virtual void checkSupportsCppAutogradFunction() const = 0;
  virtual void checkSupportsInplaceRequiresGrad() const = 0;
  virtual void checkSupportsRetainGrad() const = 0;
};

// returns deepcopy of the functorch tls
TORCH_API std::unique_ptr<FuncTorchTLSBase> getCopyOfFuncTorchTLS();

// sets the functorch tls. always does a deep copy.
TORCH_API void setFuncTorchTLS(
    const std::shared_ptr<const FuncTorchTLSBase>& state);

// get a mutable reference to the functorch tls
TORCH_API std::unique_ptr<FuncTorchTLSBase>& functorchTLSAccessor();

} // namespace at::functorch
