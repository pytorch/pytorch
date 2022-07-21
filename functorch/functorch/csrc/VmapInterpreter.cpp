#include <functorch/csrc/VmapInterpreter.h>
#include <functorch/csrc/DynamicLayer.h>

namespace at { namespace functorch {

void VmapInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  DispatchKeySet exclude = keysToExcludeWhenEnteringDynamicLayer(TransformType::Vmap);
  setup_dispatch_key_tls(exclude, DispatchKeySet(kVmapModeKey));
  op.callBoxed(stack);
}

void VmapInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // Re-dispatch
  if (getDynamicLayerStack().size() == 0) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);
}

}} // namespace at::functorch
