#include <ATen/functorch/FunctionalizeInterpreter.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/FunctionalTensorWrapper.h>

namespace at::functorch {

static void sanityCheckNotFunctional(const c10::OperatorHandle& op, torch::jit::Stack* stack, size_t num_args) {
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {
        TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(tensor));
        return tensor;
      });
}

void FunctionalizeInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // We always want to call the functionalization kernels if functionalize() is on the layer stack.
  // It's the responsibility of the functionalization kernel to no-op and redispatch
  // if none of the input tensors are functional.
  setup_dispatch_key_tls(TransformType::Functionalize, DispatchKeySet(DispatchKey::Functionalize));
  auto functionalization_add_back_views = functionalizeAddBackViews();
  // We have some side-car TLS that we can set to toggle the functionaliation behavior.
  // If set, then we functionalization will only remove mutations, instead of
  // removing both mutations AND view operators.
  at::functionalization::impl::FunctionalizationReapplyViewsGuard functional_guard(functionalization_add_back_views);

  op.callBoxed(stack);

  auto ret_size = op.schema().returns().size();
  foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(),
    [&](const Tensor& tensor) {
      if (at::functionalization::impl::isFunctionalTensor(tensor)) {
        auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
        // Functorch is responsible for setting the level on the wrapper, since we don't
        // have that info available in core (for now).
        // We could just "propagate" the level from the input tensors inside of the functionalize kernels,
        // but unfortunately we can't do that for factory operators.
        wrapper->set_level(level());
      }
      return tensor;
    }
  );
}

void FunctionalizeInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  // For now, we don't support nested functionalization calls.
  // This check just enforces that - after the functionalize kernel runs
  // and we hit the BackModeFallback, we'll have unwrapped our FunctionalTensors
  // so we can check that the unwrapped thing is not another (nested) FunctionalTensor.
  auto args_size = op.schema().arguments().size();
  sanityCheckNotFunctional(op, stack, args_size);

  // Re-dispatch
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);

  auto ret_size = op.schema().returns().size();
  sanityCheckNotFunctional(op, stack, ret_size);
}

} // namespace at::functorch
