#pragma once

#include <c10/macros/Macros.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at {
namespace impl {

// TorchDispatchOverride represents the information necessary to override
// __torch_dispatch__. Concretely it is a Tensor, but the relevant information is:
// - which PyInterpreter we are talking about
// - which __torch_dispatch__ function.
//
// Note [TorchDispatchOverride indirection]
// PyInterpreter is defined in TensorImpl.h. In order to dispatch to
// __torch_dispatch__, we must go through the PyInterpreter (because there may
// be multiple because of torchdeploy).
//
// Now the problem with PyInterpreter being in TensorImpl.h is that c10::Tensor
// hasn't been defined yet and so we define a struct that has a Tensor member.
// So we do this indirection where we define TorchDispatchOverride in
// TensorImpl.h and subclass it here.
struct TORCH_API TorchDispatchOverrideImpl : TorchDispatchOverride {
  TorchDispatchOverrideImpl(const Tensor& tensor) : tensor_(tensor) {}
  const Tensor& tensor() const override {
    return tensor_;
  }
 private:
  Tensor tensor_;
};

// When set, all factory operators will use the TLS TorchDispatchOverride
// to dispatch. (A factory operator is an operator that
// has no Tensors arguments but returns a Tensor, e.g. torch.empty).
//
// The mechanism by which this happens is the PythonMode dispatch key.
// When the TorchDispatchOverride is set, DispatchKey::PythonMode TLS is enabled.
// DispatchKey::PythonMode has a fallthrough kernel but RegisterPythonMode.cpp
// registers kernels for PythonMode that call `dispatchFactoryToPython` (see
// below for more details).
struct TORCH_API PythonMode {
  // Use this tensor's PyInterpreter and __torch_dispatch__ as the TorchDispatchOverride
  static void set_torch_dispatch_with_tensor(const Tensor& tensor);

  // Removes the TorchDispatchOverride TLS
  static void reset_torch_dispatch();

  // Set and get the TorchDispatchOverride
  static void set_torch_dispatch(const TorchDispatchOverrideImpl& dispatch_override);
  static const optional<TorchDispatchOverrideImpl>& get_torch_dispatch();
};

// This is a boxed kernel that (1) assumes `op` and `stack` are for a
// factory function and (2) dispatches to the __torch_dispatch__ recorded in
// the TorchDispatchOverride TLS.
TORCH_API void dispatchFactoryToPython(const c10::OperatorHandle& op, torch::jit::Stack* stack);

} // namespace impl
} // namespace at
