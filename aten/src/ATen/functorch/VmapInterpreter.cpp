#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/DynamicLayer.h>

namespace at::functorch {

void VmapInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  setup_dispatch_key_tls(TransformType::Vmap, DispatchKeySet(DispatchKey::FuncTorchVmapMode));
  op.callBoxed(stack);
}

void VmapInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  // Re-dispatch
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);
}

} // namespace at::functorch
